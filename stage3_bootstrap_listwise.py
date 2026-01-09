#!/usr/bin/env python3
"""
Stage 3: Batch bootstrap + listwise ranking + aggregation (full version)
Processes every query, runs bootstrap listwise ranking, and saves detailed/intermediate outputs.
"""

import json
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
import math
import random
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
import numpy as np

# API configuration (read from environment; no hardcoded secrets)
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini-2025-08-07")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required for Stage 3.")

# Create an OpenAI client per thread
def create_client():
    return OpenAI(api_key=API_KEY)

# File paths
STAGE2_DETAILED_FILE = "../stage2/stage2_results_detailed.jsonl"
CHUNK_CARDS_DIR = Path("../data/finbenchqa/filtered_data")
OUTPUT_DIR = Path(".")
OUTPUT_DETAILED = OUTPUT_DIR / "stage3_results_detailed.jsonl"
OUTPUT_INTERMEDIATE = OUTPUT_DIR / "stage3_intermediate_results.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "stage3_results.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "stage3_summary.json"

# Concurrency
MAX_WORKERS = 1000  # API concurrency

# Stage 3 parameters
MIN_GROUP_SIZE = 15
MAX_GROUP_SIZE = 25
MAX_ROUNDS = 5  # maximum rounds
CONVERGENCE_THRESHOLD = 0.9  # Jaccard similarity threshold
FINAL_TOP_K = 25  # final Top-K


def determine_group_size_and_rounds(
    num_candidates: int,
    min_group_size: int = 15,
    max_group_size: int = 25,
    max_rounds: int = 5
) -> Tuple[int, int]:
    """Dynamically choose group size and initial rounds."""
    if num_candidates <= min_group_size:
        group_size = num_candidates
        initial_rounds = 1
    elif num_candidates <= max_group_size * 2:
        group_size = max(min_group_size, num_candidates // 2)
        initial_rounds = 2
    else:
        group_size = max_group_size
        initial_rounds = min(max_rounds, math.ceil(num_candidates / group_size))

    return group_size, initial_rounds


def random_shuffle_and_group(
    candidates: List[Dict],
    group_size: int
) -> List[List[Dict]]:
    """Shuffle candidates and split into groups covering all candidates."""
    shuffled = candidates.copy()
    random.shuffle(shuffled)

    groups = []
    for i in range(0, len(shuffled), group_size):
        groups.append(shuffled[i:i + group_size])

    return groups


def create_listwise_ranking_prompt(
    question_text: str,
    group_candidates: List[Dict],
    chunk_cards_map: Dict
) -> str:
    """Build prompt for listwise ranking within a group."""
    group_cards_info = []
    for candidate in group_candidates:
        chunk_id = candidate['chunk_id']
        chunk_card = chunk_cards_map.get(chunk_id, {})

        card_copy = {k: v for k, v in chunk_card.items() if k != 'boilerplate_flag'}

        card_info = {
            'chunk_id': chunk_id,
            'card': card_copy
        }
        group_cards_info.append(card_info)

    prompt = f"""You are a financial document analysis expert. Your task is to rank a group of evidence chunks from most relevant to least relevant for answering a specific question.

QUESTION:
{question_text}

CANDIDATE CHUNKS IN THIS GROUP:
You need to rank all {len(group_candidates)} candidate chunks in order of relevance.

CANDIDATE CHUNKS WITH THEIR CARDS:
{json.dumps(group_cards_info, indent=2, ensure_ascii=False)}

RANKING CRITERIA:
1. **Metric Matching**: Check if the chunk contains financial metrics relevant to the question (e.g., Revenue, COGS, Net Income).
2. **Temporal Matching**: Check if the chunk's temporal data aligns with the question's time requirements (e.g., FY2023, Q1 2024).
3. **Scope Matching**: Check if the chunk's scope (company-wide, subsidiary, segment, region) matches the question's requirements.
4. **Content Type**: For quantitative questions, prioritize chunks with table data. For qualitative questions, text data is acceptable.
5. **Relevance**: Consider the summary, financial_metrics, and overall relevance to the question.

CRITICAL CONSTRAINTS:
- **DO NOT access the original chunk text** - you can only use the card information provided
- **DO NOT perform numerical calculations** - only rank based on qualitative relevance
- **DO NOT assign absolute scores** - only provide relative ranking
- You must rank ALL chunks in the group, from most relevant to least relevant

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
  "ranked_chunks": [
    {{
      "chunk_id": "chunk_id_here",
      "rank": 1,
      "reason": "Brief reason why this chunk is ranked at this position (1-2 sentences)"
    }},
    {{
      "chunk_id": "chunk_id_here",
      "rank": 2,
      "reason": "Brief reason..."
    }},
    ...
  ]
}}

The "ranked_chunks" array should contain ALL {len(group_candidates)} chunks, ordered from rank 1 (most relevant) to rank {len(group_candidates)} (least relevant).
Each chunk should have:
- chunk_id: The ID of the chunk
- rank: The rank position (1 = most relevant, {len(group_candidates)} = least relevant)
- reason: A brief explanation (1-2 sentences) for why this chunk is ranked at this position

IMPORTANT:
- Rank ALL chunks - do not skip any
- Provide a complete ranking from 1 to {len(group_candidates)}
- Base your ranking purely on the card information and relevance to the question
- Do not use any external scores or metadata - make your own judgment
- The ranking should be relative within this group only

Now analyze the candidate chunks and return the JSON with your complete ranking:"""

    return prompt


def listwise_rank_group_with_agent(
    group_candidates: List[Dict],
    question_text: str,
    chunk_cards_map: Dict,
    client: OpenAI
) -> List[Dict]:
    """Run agent-based listwise ranking for a group."""
    prompt = create_listwise_ranking_prompt(question_text, group_candidates, chunk_cards_map)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        response_text = completion.choices[0].message.content

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            agent_result = json.loads(json_str)

            ranked_chunks = agent_result.get('ranked_chunks', [])

            if len(ranked_chunks) != len(group_candidates):
                chunk_id_to_candidate = {c['chunk_id']: c for c in group_candidates}
                ranked_chunk_ids = {r.get('chunk_id') for r in ranked_chunks}
                missing_ids = set(chunk_id_to_candidate.keys()) - ranked_chunk_ids

                max_rank = max((r.get('rank', 0) for r in ranked_chunks), default=0)
                for missing_id in missing_ids:
                    max_rank += 1
                    ranked_chunks.append({
                        'chunk_id': missing_id,
                        'rank': max_rank,
                        'reason': 'Not ranked by agent (fallback)'
                    })

            result = []
            chunk_id_to_candidate = {c['chunk_id']: c for c in group_candidates}

            for ranked in ranked_chunks:
                chunk_id = ranked.get('chunk_id')
                if chunk_id in chunk_id_to_candidate:
                    candidate = chunk_id_to_candidate[chunk_id]
                    result.append({
                        **candidate,
                        'rank': ranked.get('rank', len(group_candidates)),
                        'ranking_reason': ranked.get('reason', '')
                    })

            result.sort(key=lambda x: x.get('rank', len(group_candidates)))
            return result
        else:
            return [
                {**c, 'rank': idx + 1, 'ranking_reason': 'Agent parsing failed (fallback)'}
                for idx, c in enumerate(group_candidates)
            ]

    except Exception as e:
        return [
            {**c, 'rank': idx + 1, 'ranking_reason': f'Agent error: {str(e)}'}
            for idx, c in enumerate(group_candidates)
        ]


def compute_normalized_borda_scores(
    all_round_results: List[Dict]
) -> Dict[str, Dict]:
    """Compute normalized Borda scores across all rounds/groups."""
    chunk_scores = defaultdict(lambda: {'total_score': 0.0, 'appearances': 0})

    for round_result in all_round_results:
        for group_result in round_result['groups']:
            group_size = group_result['group_size']
            ranked_chunks = group_result['ranked_chunks']

            for ranked_chunk in ranked_chunks:
                chunk_id = ranked_chunk['chunk_id']
                rank = ranked_chunk.get('rank', group_size)

                raw_score = group_size - rank
                if group_size > 1:
                    normalized_score = raw_score / (group_size - 1)
                else:
                    normalized_score = 1.0

                chunk_scores[chunk_id]['total_score'] += normalized_score
                chunk_scores[chunk_id]['appearances'] += 1

    return dict(chunk_scores)


def compute_stability_metrics(
    all_round_results: List[Dict],
    chunk_scores: Dict[str, Dict],
    top_t: int = 5
) -> Dict[str, Dict]:
    """Compute stability metrics across rounds."""
    stability_metrics = defaultdict(lambda: {
        'top_t_frequency': 0,
        'ranks': [],
        'round_ranks': []
    })

    for round_result in all_round_results:
        round_num = round_result['round']

        round_chunk_scores = defaultdict(float)
        for group_result in round_result['groups']:
            group_size = group_result['group_size']
            ranked_chunks = group_result['ranked_chunks']

            for ranked_chunk in ranked_chunks:
                chunk_id = ranked_chunk['chunk_id']
                rank = ranked_chunk.get('rank', group_size)

                raw_score = group_size - rank
                if group_size > 1:
                    normalized_score = raw_score / (group_size - 1)
                else:
                    normalized_score = 1.0

                round_chunk_scores[chunk_id] += normalized_score

        sorted_chunks = sorted(round_chunk_scores.items(), key=lambda x: x[1], reverse=True)
        round_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(sorted_chunks)}

        for chunk_id, temp_rank in round_ranks.items():
            stability_metrics[chunk_id]['ranks'].append(temp_rank)
            stability_metrics[chunk_id]['round_ranks'].append({
                'round': round_num,
                'rank': temp_rank
            })

            if temp_rank <= top_t:
                stability_metrics[chunk_id]['top_t_frequency'] += 1

    final_metrics = {}
    for chunk_id, metrics in stability_metrics.items():
        ranks = metrics['ranks']
        if ranks:
            final_metrics[chunk_id] = {
                'top_t_frequency': metrics['top_t_frequency'],
                'rank_variance': float(np.var(ranks)) if len(ranks) > 1 else 0.0,
                'avg_rank': float(np.mean(ranks)),
                'min_rank': int(min(ranks)),
                'max_rank': int(max(ranks)),
                'round_ranks': metrics['round_ranks']
            }
        else:
            final_metrics[chunk_id] = {
                'top_t_frequency': 0,
                'rank_variance': 0.0,
                'avg_rank': float('inf'),
                'min_rank': None,
                'max_rank': None,
                'round_ranks': []
            }

    return final_metrics


def compute_jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def check_convergence(
    all_round_results: List[Dict],
    chunk_scores: Dict[str, Dict],
    top_k: int,
    threshold: float = 0.9
) -> Tuple[bool, float]:
    """Check convergence via Jaccard similarity between consecutive Top-K sets."""
    if len(all_round_results) < 2:
        return False, 0.0

    last_two_rounds = all_round_results[-2:]
    top_k_sets = []

    for round_result in last_two_rounds:
        round_chunk_scores = defaultdict(float)
        for group_result in round_result['groups']:
            group_size = group_result['group_size']
            ranked_chunks = group_result['ranked_chunks']

            for ranked_chunk in ranked_chunks:
                chunk_id = ranked_chunk['chunk_id']
                rank = ranked_chunk.get('rank', group_size)

                raw_score = group_size - rank
                if group_size > 1:
                    normalized_score = raw_score / (group_size - 1)
                else:
                    normalized_score = 1.0

                round_chunk_scores[chunk_id] += normalized_score

        sorted_chunks = sorted(round_chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_chunk_ids = {chunk_id for chunk_id, _ in sorted_chunks[:top_k]}
        top_k_sets.append(top_k_chunk_ids)

    jaccard = compute_jaccard_similarity(top_k_sets[0], top_k_sets[1])
    is_converged = jaccard >= threshold

    return is_converged, jaccard


def load_chunk_cards_for_doc(doc_id: str) -> Dict:
    """Load chunk cards for a given document."""
    card_file = CHUNK_CARDS_DIR / f"upgraded_chunk_cards_{doc_id}.json"

    if not card_file.exists():
        return {}

    try:
        with open(card_file, 'r', encoding='utf-8') as f:
            chunk_cards_data = json.load(f)

        chunk_cards_map = {}
        for item in chunk_cards_data:
            chunk_uid = item.get('chunk_uid')
            card = item.get('card', {})
            if chunk_uid and card:
                chunk_cards_map[chunk_uid] = card

        return chunk_cards_map
    except Exception:
        return {}


def process_query_stage3(
    query_result: Dict,
    chunk_cards_map: Dict,
    min_group_size: int = 15,
    max_group_size: int = 25,
    max_rounds: int = 5,
    convergence_threshold: float = 0.9,
    final_top_k: int = 25
) -> Dict:
    """Execute Stage 3 for a single query."""
    query_id = query_result['query_id']
    question_text = query_result['question_text']
    candidates = query_result['final_candidates']

    client = create_client()

    group_size, _ = determine_group_size_and_rounds(
        len(candidates),
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        max_rounds=max_rounds
    )

    all_round_results = []

    for round_num in range(1, max_rounds + 1):
        groups = random_shuffle_and_group(candidates, group_size)

        round_groups = []
        for group_idx, group_candidates in enumerate(groups):
            ranked_chunks = listwise_rank_group_with_agent(
                group_candidates,
                question_text,
                chunk_cards_map,
                client
            )

            round_groups.append({
                'group_idx': group_idx,
                'group_size': len(group_candidates),
                'ranked_chunks': ranked_chunks
            })

        round_result = {
            'round': round_num,
            'groups': round_groups
        }
        all_round_results.append(round_result)

        chunk_scores = compute_normalized_borda_scores(all_round_results)

        if round_num >= 2:
            is_converged, _ = check_convergence(
                all_round_results,
                chunk_scores,
                top_k=final_top_k,
                threshold=convergence_threshold
            )

            if is_converged:
                break

    final_chunk_scores = compute_normalized_borda_scores(all_round_results)

    stability_metrics = compute_stability_metrics(
        all_round_results,
        final_chunk_scores,
        top_t=5
    )

    sorted_chunks = sorted(
        final_chunk_scores.items(),
        key=lambda x: x[1]['total_score'],
        reverse=True
    )

    final_ranked_candidates = []
    for rank, (chunk_id, score_info) in enumerate(sorted_chunks, 1):
        original_candidate = next(
            (c for c in candidates if c['chunk_id'] == chunk_id),
            {'chunk_id': chunk_id}
        )

        final_ranked_candidates.append({
            **original_candidate,
            'final_rank': rank,
            'aggregated_score': score_info['total_score'],
            'appearances': score_info['appearances'],
            'stability_metrics': stability_metrics.get(chunk_id, {})
        })

    final_top_k_candidates = final_ranked_candidates[:final_top_k]

    return {
        'query_id': query_id,
        'question_text': question_text,
        'doc_id': query_result.get('doc_id'),
        'stage2_candidate_count': len(candidates),
        'num_rounds': len(all_round_results),
        'group_size': group_size,
        'final_ranked_candidates': final_ranked_candidates,
        'final_top_k_candidates': final_top_k_candidates,
        'final_top_k_count': len(final_top_k_candidates),
        'all_round_results': all_round_results,
        'chunk_scores': final_chunk_scores,
        'stability_metrics': stability_metrics
    }


def process_single_query_wrapper(args):
    """Wrapper for concurrent processing."""
    query_result, chunk_cards_map, min_group_size, max_group_size, max_rounds, convergence_threshold, final_top_k = args

    try:
        result = process_query_stage3(
            query_result,
            chunk_cards_map,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            max_rounds=max_rounds,
            convergence_threshold=convergence_threshold,
            final_top_k=final_top_k
        )
        return result
    except Exception as e:
        return {
            'query_id': query_result.get('query_id', 'unknown'),
            'error': str(e)
        }


def compute_summary(all_results: List[Dict]) -> Dict:
    """Compute global summary statistics (gold-free)."""
    valid_results = [r for r in all_results if 'error' not in r]

    if not valid_results:
        return {'error': 'No valid results'}

    num_queries = len(valid_results)

    avg_final_candidates = sum(r['final_top_k_count'] for r in valid_results) / num_queries
    avg_rounds = sum(r['num_rounds'] for r in valid_results) / num_queries

    summary = {
        'num_queries': num_queries,
        'avg_final_top_k_count': round(avg_final_candidates, 2),
        'avg_rounds': round(avg_rounds, 2)
    }

    return summary


def main():
    print("=" * 70)
    print("Stage 3: Bootstrap + listwise ranking + aggregation")
    print("=" * 70)
    print(f"API concurrency: {MAX_WORKERS}")
    print(f"Group size range: {MIN_GROUP_SIZE}-{MAX_GROUP_SIZE}")
    print(f"Max rounds: {MAX_ROUNDS}, convergence threshold: {CONVERGENCE_THRESHOLD}")
    print(f"Final Top-K: {FINAL_TOP_K}")
    print()

    # 1. Load Stage 2 results
    print("1. Loading Stage 2 candidate results...")
    stage2_results = []
    with open(STAGE2_DETAILED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            if 'error' not in result:
                stage2_results.append(result)

    print(f"   Loaded {len(stage2_results)} queries across documents")

    # 2. Group by document and load chunk cards
    print("\n2. Loading chunk cards (grouped by document)...")
    doc_to_queries = defaultdict(list)
    for result in stage2_results:
        doc_id = result.get('doc_id')
        if doc_id:
            doc_to_queries[doc_id].append(result)

    print(f"   Found {len(doc_to_queries)} documents")
    chunk_cards_cache = {}

    for doc_id, queries in doc_to_queries.items():
        if doc_id not in chunk_cards_cache:
            print(f"   Loading cards for {doc_id}...")
            chunk_cards_cache[doc_id] = load_chunk_cards_for_doc(doc_id)
            print(f"     → Loaded {len(chunk_cards_cache[doc_id])} chunk cards")

    # 3. Prepare tasks
    print("\n3. Preparing tasks...")
    all_tasks = []
    for result in stage2_results:
        doc_id = result.get('doc_id')
        chunk_cards_map = chunk_cards_cache.get(doc_id, {})

        all_tasks.append((
            result,
            chunk_cards_map,
            MIN_GROUP_SIZE,
            MAX_GROUP_SIZE,
            MAX_ROUNDS,
            CONVERGENCE_THRESHOLD,
            FINAL_TOP_K
        ))

    print(f"   Total tasks: {len(all_tasks)}")

    # 4. Concurrent processing
    print(f"\n4. Starting concurrent processing ({MAX_WORKERS} workers)...")
    all_results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_query_wrapper, args): args[0]['query_id']
            for args in all_tasks
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            all_results.append(result)

            query_id = result.get('query_id', 'unknown')
            if 'error' in result:
                print(f"  [{completed}/{len(all_tasks)}] {query_id}: ✗ Error - {result['error']}")
            else:
                rounds = result.get('num_rounds', 0)
                print(f"  [{completed}/{len(all_tasks)}] {query_id}: ✓ Top-K {result.get('final_top_k_count', 0)}, rounds {rounds}")

            if completed % 10 == 0:
                elapsed = time.time() - start_time
                print(f"    Progress: {completed}/{len(all_tasks)} ({completed/len(all_tasks)*100:.1f}%), "
                      f"elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\nProcessing complete! Elapsed: {elapsed:.1f}s")

    # 5. Save detailed results
    print(f"\n5. Saving detailed results to {OUTPUT_DETAILED}...")
    with open(OUTPUT_DETAILED, 'w', encoding='utf-8') as f:
        for result in all_results:
            if 'error' in result:
                continue

            simplified_result = {
                'query_id': result.get('query_id'),
                'question_text': result.get('question_text'),
                'doc_id': result.get('doc_id'),
                'stage2_candidate_count': result.get('stage2_candidate_count'),
                'num_rounds': result.get('num_rounds'),
                'group_size': result.get('group_size'),
                'final_top_k_count': result.get('final_top_k_count'),
                'final_ranked_candidates': [
                    {
                        'chunk_id': c['chunk_id'],
                        'final_rank': c['final_rank'],
                        'aggregated_score': c['aggregated_score'],
                        'appearances': c['appearances'],
                        'top_t_frequency': c.get('stability_metrics', {}).get('top_t_frequency', 0),
                        'rank_variance': c.get('stability_metrics', {}).get('rank_variance', 0.0),
                        'avg_rank': c.get('stability_metrics', {}).get('avg_rank', 0.0)
                    }
                    for c in result.get('final_ranked_candidates', [])
                ],
                'final_top_k_candidates': [
                    {
                        'chunk_id': c['chunk_id'],
                        'final_rank': c['final_rank'],
                        'aggregated_score': c['aggregated_score']
                    }
                    for c in result.get('final_top_k_candidates', [])
                ]
            }
            f.write(json.dumps(simplified_result, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved {len([r for r in all_results if 'error' not in r])} records")

    # 6. Save intermediate trace
    print(f"\n6. Saving intermediate trace to {OUTPUT_INTERMEDIATE}...")
    with open(OUTPUT_INTERMEDIATE, 'w', encoding='utf-8') as f:
        for result in all_results:
            if 'error' in result:
                continue

            intermediate_result = {
                'query_id': result.get('query_id'),
                'question_text': result.get('question_text'),
                'doc_id': result.get('doc_id'),
                'num_rounds': result.get('num_rounds'),
                'group_size': result.get('group_size'),
                'all_round_results': result.get('all_round_results', []),
                'chunk_scores': {
                    chunk_id: {
                        'total_score': info['total_score'],
                        'appearances': info['appearances']
                    }
                    for chunk_id, info in result.get('chunk_scores', {}).items()
                },
                'stability_metrics': result.get('stability_metrics', {})
            }
            f.write(json.dumps(intermediate_result, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved {len([r for r in all_results if 'error' not in r])} intermediate records")

    # 7. Compute and save summary
    print(f"\n7. Computing global summary...")
    summary = compute_summary(all_results)

    print(f"Saving summary to {OUTPUT_SUMMARY}...")
    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved")

    # 8. Save CSV
    print(f"\n8. Saving CSV to {OUTPUT_CSV}...")
    csv_data = []
    for result in all_results:
        if 'error' in result:
            continue
        csv_data.append({
            'query_id': result['query_id'],
            'doc_id': result.get('doc_id'),
            'stage2_candidate_count': result['stage2_candidate_count'],
            'num_rounds': result['num_rounds'],
            'group_size': result['group_size'],
            'final_top_k_count': result['final_top_k_count']
        })

    df = pd.DataFrame(csv_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"   ✓ Saved {len(csv_data)} records")

    # 9. Print summary
    print("\n" + "=" * 70)
    print("Global summary")
    print("=" * 70)
    print(f"Queries: {summary['num_queries']}")
    print(f"Average final Top-K count: {summary['avg_final_top_k_count']:.1f}")
    print(f"Average rounds: {summary['avg_rounds']:.1f}")

    print("\n" + "=" * 70)
    print("✓ Stage 3 completed")
    print("=" * 70)


if __name__ == "__main__":
    main()

