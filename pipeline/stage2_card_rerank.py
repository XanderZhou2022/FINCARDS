#!/usr/bin/env python3
"""
Stage 2: Batch dynamic grouping, filtering, and reranking (full version)
Processes every query across all documents, performs agent-based group filtering,
and saves detailed/intermediate outputs.
"""

import json
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import math
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

# API configuration (read from environment; no hardcoded secrets)
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini-2025-08-07")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required for Stage 2.")

# Create shared OpenAI client
client = OpenAI(api_key=API_KEY)

# File paths
STAGE1_DETAILED_FILE = "../stage1/stage1_candidates_detailed.jsonl"
CHUNK_CARDS_DIR = Path("../data/finbenchqa/filtered_data")
OUTPUT_DIR = Path(".")
OUTPUT_DETAILED = OUTPUT_DIR / "stage2_results_detailed.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "stage2_summary.json"
OUTPUT_CSV = OUTPUT_DIR / "stage2_results.csv"
OUTPUT_INTERMEDIATE = OUTPUT_DIR / "stage2_intermediate_results.jsonl"  # intermediate trace

# Concurrency
MAX_WORKERS = 200  # concurrent API calls

# Processing parameters
GROUP_SIZE = 25
MIN_K_PER_GROUP = 8
MAX_K_PER_GROUP = 15
FINAL_TOP_N = 50


def round_robin_group(candidates: List[Dict], group_size: int = 25) -> List[List[Dict]]:
    """Round-robin grouping so each group mixes high/medium/low BM25 scores."""
    if not candidates:
        return []
    
    num_groups = math.ceil(len(candidates) / group_size)
    groups = [[] for _ in range(num_groups)]
    
    # Round-robin assignment across groups
    for idx, candidate in enumerate(candidates):
        group_idx = idx % num_groups
        groups[group_idx].append(candidate)
    
    return groups


def create_group_filter_prompt(
    question_text: str,
    group_candidates: List[Dict],
    chunk_cards_map: Dict,
    min_k: int = 8,
    max_k: int = 15
) -> str:
    """Build prompt for the agent to filter a single group."""
    # Prepare card info without boilerplate flags
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
    
    # Adjust bounds if the group is smaller
    actual_max_k = min(max_k, len(group_candidates))
    actual_min_k = min(min_k, len(group_candidates))
    
    # Identify quotas based on question type
    question_lower = question_text.lower()
    has_temporal_trend = any(keyword in question_lower for keyword in [
        'trend', 'over time', 'across', 'multiple', 'period', 'quarter', 'year', 
        'comparison', 'compare', 'change', 'growth', 'decline', 'increase', 'decrease',
        'historical', 'previous', 'prior', 'consecutive', 'sequential'
    ])
    
    is_definition_explanation = any(keyword in question_lower for keyword in [
        'what is', 'what are', 'define', 'definition', 'meaning', 'explain', 
        'explanation', 'how is', 'how are', 'measure', 'measurement', 'basis',
        'accounting policy', 'accounting method', 'policy', 'methodology'
    ])
    
    # Build quota instructions
    coverage_quota_text = ""
    if has_temporal_trend:
        coverage_quota_text += """**TIME TREND/MULTI-PERIOD QUESTION DETECTED**
- You MUST select at least 1-2 chunks that are TABLE-BASED (tables present in card) AND have CLEAR temporal dimensions (e.g., multiple periods, time series data).
- These table chunks are CRITICAL for answering temporal/trend questions, even if they have lower individual relevance scores.

"""
    if is_definition_explanation:
        coverage_quota_text += """**DEFINITION/EXPLANATION QUESTION DETECTED**
- You MUST select at least 1 chunk that contains definition, measurement basis, or accounting policy information.
- This chunk is REQUIRED even if it has boilerplate_flag=true, as definitions are often in standard sections.

"""
    
    prompt = f"""You are a financial document analysis expert. Your task is to select the most relevant evidence chunks from a group of candidates to answer a specific question.

QUESTION:
{question_text}

CANDIDATE CHUNKS IN THIS GROUP:
You need to analyze {len(group_candidates)} candidate chunks and select between {actual_min_k} and {actual_max_k} most relevant ones.

CANDIDATE CHUNKS WITH THEIR CARDS:
{json.dumps(group_cards_info, indent=2, ensure_ascii=False)}

SELECTION CRITERIA:
1. **Metric Matching**: Check if the chunk contains financial metrics relevant to the question (e.g., Revenue, COGS, Net Income).
2. **Temporal Matching**: Check if the chunk's temporal data aligns with the question's time requirements (e.g., FY2023, Q1 2024).
3. **Scope Matching**: Check if the chunk's scope (company-wide, subsidiary, segment, region) matches the question's requirements.
4. **Content Type**: For quantitative questions, prioritize chunks with table data. For qualitative questions, text data is acceptable.
5. **Relevance**: Consider the summary, financial_metrics, and overall relevance to the question.

MANDATORY COVERAGE QUOTAS (MUST ENFORCE):
{coverage_quota_text if coverage_quota_text else "(No specific coverage quotas for this question type)"}

CRITICAL WARNING ABOUT TABLES:
- **Table chunks require EXTRA CAUTION**: If a chunk contains tables, especially CONTINUOUS/SEQUENTIAL tables (multiple related tables in sequence), you must carefully verify:
  1. The table structure and headers match the question's requirements
  2. The temporal coverage (time periods) is appropriate
  3. The metrics in the table are relevant to the question
  4. For continuous tables, ensure you understand the relationship between adjacent tables
- **DO NOT blindly select table chunks** just because they contain tables - verify their actual relevance to the specific question
- **Sequential/continuous tables** can be misleading if not properly analyzed - check if they represent the same data across different periods or different aspects
- If a table chunk seems relevant but you're uncertain, prefer chunks with clear, unambiguous table structures

NOTE: You should evaluate chunks based purely on their card information (financial_metrics, temporal_data, scope, tables, summary, etc.) and their relevance to the question. Do not rely on any external ranking scores - make your own judgment based on the content.

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
  "selected_chunks": [
    {{
      "chunk_id": "chunk_id_here",
      "selection_reasons": [
        "Reason 1: ...",
        "Reason 2: ...",
        ...
      ],
      "relevance_score": <score from 0-100>
    }}
  ]
}}

The "selected_chunks" array should contain between {actual_min_k} and {actual_max_k} chunks, ordered by relevance (most relevant first).
Each chunk should have:
- chunk_id: The ID of the selected chunk
- selection_reasons: An array of reasons explaining why this chunk was selected (should reference the selection criteria and coverage quotas)
- relevance_score: A score from 0-100 indicating how relevant this chunk is to the question

IMPORTANT:
- Select between {actual_min_k} and {actual_max_k} chunks (you can choose the exact number based on relevance)
- If there are fewer than {actual_min_k} relevant chunks, select all relevant ones (but at least try to find {actual_min_k})
- If there are many highly relevant chunks, you can select up to {actual_max_k} chunks
- **ENFORCE THE MANDATORY COVERAGE QUOTAS ABOVE** - these are hard requirements, not suggestions
- Order them by relevance (highest relevance first)
- Provide clear, specific reasons for each selection, especially indicating which chunks satisfy the coverage quotas
- Consider all selection criteria
- Evaluate chunks purely based on their content and relevance, not on any external flags or metadata
- **Be especially careful with table chunks** - verify their actual relevance before selecting

Now analyze the candidate chunks and return the JSON with your selections:"""
    
    return prompt


def filter_group_candidates_with_agent(
    group_candidates: List[Dict],
    question_text: str,
    chunk_cards_map: Dict,
    min_k: int = 8,
    max_k: int = 15
) -> List[Dict]:
    """Run the agent to filter a group and return 8-15 items (dynamic)."""
    # Build prompt
    prompt = create_group_filter_prompt(question_text, group_candidates, chunk_cards_map, min_k, max_k)
    
    # Call the agent
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
        
        # Try to parse JSON from the agent response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            agent_result = json.loads(json_str)
            
            selected_chunks = agent_result.get('selected_chunks', [])
            
            # Validate that the return count is within expected bounds
            num_selected = len(selected_chunks)
            if num_selected < min_k:
                pass  # Allow fewer than min_k if genuinely few relevant chunks
            elif num_selected > max_k:
                selected_chunks = selected_chunks[:max_k]
            
            # Convert agent output into the standard candidate format
            result = []
            chunk_id_to_candidate = {c['chunk_id']: c for c in group_candidates}
            
            for selected in selected_chunks:
                chunk_id = selected.get('chunk_id')
                if chunk_id in chunk_id_to_candidate:
                    candidate = chunk_id_to_candidate[chunk_id]
                    result.append({
                        **candidate,
                        'relevance_score': selected.get('relevance_score', 0),
                        'selection_reasons': {
                            'chunk_id': chunk_id,
                            'reasons': selected.get('selection_reasons', []),
                            'agent_score': selected.get('relevance_score', 0)
                        }
                    })
            
            return result
        else:
            print("    ⚠️  Warning: could not extract JSON from agent response")
            return []
            
    except Exception as e:
        print(f"    ✗ Agent call failed: {str(e)}")
        return []


def process_query_stage2(
    query_result: Dict,
    chunk_cards_map: Dict,
    group_size: int = 25,
    min_k_per_group: int = 8,
    max_k_per_group: int = 15,
    final_top_n: int = 50
) -> Dict:
    """Run Stage 2 for a single query (group selection + reranking)."""
    query_id = query_result['query_id']
    question_text = query_result['question_text']
    candidates = query_result['candidates']  # already sorted by BM25
    
    # 1. Round-robin grouping
    groups = round_robin_group(candidates, group_size=group_size)
    
    # 2. Agent filtering per group
    all_selected = []
    group_results = []  # capture intermediate group info
    
    for group_idx, group_candidates in enumerate(groups):
        selected = filter_group_candidates_with_agent(
            group_candidates,
            question_text,
            chunk_cards_map,
            min_k=min_k_per_group,
            max_k=max_k_per_group
        )
        all_selected.extend(selected)
        
        # Save intermediate group info
        group_results.append({
            'group_idx': group_idx,
            'group_size': len(group_candidates),
            'selected_count': len(selected),
            'selected_chunk_ids': [s['chunk_id'] for s in selected]
        })
    
    # 3. Deduplicate by chunk_id
    seen = set()
    unique_selected = []
    for candidate in all_selected:
        chunk_id = candidate['chunk_id']
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_selected.append(candidate)
    
    # 4. Sort by agent-provided relevance and clip to Top-N
    unique_selected.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    final_candidates = unique_selected[:final_top_n]
    
    return {
        'query_id': query_id,
        'question_text': question_text,
        'doc_id': query_result.get('doc_id'),
        'original_candidate_count': len(candidates),
        'num_groups': len(groups),
        'selected_after_grouping': len(all_selected),
        'unique_selected': len(unique_selected),
        'final_candidate_count': len(final_candidates),
        'final_candidates': final_candidates,
        'group_results': group_results  # intermediate trace
    }


def load_chunk_cards_for_doc(doc_id: str) -> Dict:
    """Load chunk cards for a given document."""
    card_file = CHUNK_CARDS_DIR / f"upgraded_chunk_cards_{doc_id}.json"
    
    if not card_file.exists():
        print(f"  ⚠️  Warning: {card_file} does not exist")
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
    except Exception as e:
        print(f"  ✗ Failed to load {card_file}: {str(e)}")
        return {}


def process_single_query_wrapper(args):
    """Wrapper for concurrent execution."""
    query_result, chunk_cards_map, group_size, min_k_per_group, max_k_per_group, final_top_n = args
    
    try:
        result = process_query_stage2(
            query_result,
            chunk_cards_map,
            group_size=group_size,
            min_k_per_group=min_k_per_group,
            max_k_per_group=max_k_per_group,
            final_top_n=final_top_n
        )
        return result
    except Exception as e:
        return {
            'query_id': query_result.get('query_id', 'unknown'),
            'error': str(e)
        }


def compute_summary(all_results: List[Dict]) -> Dict:
    """Compute global summary statistics (without golden metrics)."""
    valid_results = [r for r in all_results if 'error' not in r]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    num_queries = len(valid_results)
    
    avg_final_candidates = sum(r['final_candidate_count'] for r in valid_results) / num_queries
    avg_unique_selected = sum(r['unique_selected'] for r in valid_results) / num_queries
    
    summary = {
        'num_queries': num_queries,
        'avg_final_candidate_count': round(avg_final_candidates, 2),
        'avg_unique_selected': round(avg_unique_selected, 2)
    }
    
    return summary


def main():
    print("=" * 70)
    print("Stage 2: Batch dynamic grouping + filtering + reranking")
    print("=" * 70)
    print(f"API concurrency: {MAX_WORKERS}")
    print(f"Group size: {GROUP_SIZE}, selection per group: {MIN_K_PER_GROUP}-{MAX_K_PER_GROUP}, final Top-N: {FINAL_TOP_N}")
    print()
    
    # 1. Load Stage 1 candidates
    print("1. Loading Stage 1 candidate results...")
    stage1_results = []
    with open(STAGE1_DETAILED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            if 'error' not in result:
                stage1_results.append(result)
    
    print(f"   Loaded {len(stage1_results)} queries")
    
    # 2. Group by document and load chunk cards
    print("\n2. Loading chunk cards (grouped by document)...")
    doc_to_queries = defaultdict(list)
    for result in stage1_results:
        doc_id = result.get('doc_id')
        if doc_id:
            doc_to_queries[doc_id].append(result)
    
    print(f"   Found {len(doc_to_queries)} documents")
    chunk_cards_cache = {}  # cache loaded cards
    
    # 3. Prepare tasks
    print("\n3. Preparing tasks...")
    all_tasks = []
    for doc_id, queries in doc_to_queries.items():
        if doc_id not in chunk_cards_cache:
            print(f"   Loading cards for {doc_id}...")
            chunk_cards_cache[doc_id] = load_chunk_cards_for_doc(doc_id)
            print(f"     → Loaded {len(chunk_cards_cache[doc_id])} chunk cards")
        
        chunk_cards_map = chunk_cards_cache[doc_id]
        
        for query_result in queries:
            all_tasks.append((
                query_result,
                chunk_cards_map,
                GROUP_SIZE,
                MIN_K_PER_GROUP,
                MAX_K_PER_GROUP,
                FINAL_TOP_N
            ))
    
    print(f"   Total tasks: {len(all_tasks)}")
    
    # 4. Concurrent processing
    print(f"\n4. Starting concurrent processing ({MAX_WORKERS} workers)...")
    all_results = []
    intermediate_results = []  # intermediate trace
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
                kept = result.get('final_candidate_count', 0)
                print(f"  [{completed}/{len(all_tasks)}] {query_id}: ✓ kept {kept} final candidates")
            
            # Save intermediate trace
            intermediate_result = {
                'query_id': result.get('query_id'),
                'timestamp': time.time(),
                'group_results': result.get('group_results', [])
            }
            intermediate_results.append(intermediate_result)
            
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                print(f"    Progress: {completed}/{len(all_tasks)} ({completed/len(all_tasks)*100:.1f}%), "
                      f"elapsed: {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"\nProcessing complete! Elapsed: {elapsed:.1f}s")

    # 5. Save detailed results
    print(f"\n5. Saving detailed results to {OUTPUT_DETAILED}...")
    with open(OUTPUT_DETAILED, 'w', encoding='utf-8') as f:
        for result in all_results:
            # Simplify candidate payloads for storage
            simplified_result = {
                'query_id': result.get('query_id'),
                'question_text': result.get('question_text'),
                'doc_id': result.get('doc_id'),
                'original_candidate_count': result.get('original_candidate_count'),
                'num_groups': result.get('num_groups'),
                'final_candidate_count': result.get('final_candidate_count'),
                'final_candidates': [
                    {
                        'chunk_id': c['chunk_id'],
                        'bm25_score': c.get('bm25_score', 0),
                        'relevance_score': c.get('relevance_score', 0),
                        'selection_reasons': c.get('selection_reasons', {}).get('reasons', [])
                        if isinstance(c.get('selection_reasons'), dict) else []
                    }
                    for c in result.get('final_candidates', [])
                ]
            }
            if 'error' in result:
                simplified_result['error'] = result['error']
            f.write(json.dumps(simplified_result, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved {len(all_results)} records")

    # 6. Save intermediate trace
    print(f"\n6. Saving intermediate trace to {OUTPUT_INTERMEDIATE}...")
    with open(OUTPUT_INTERMEDIATE, 'w', encoding='utf-8') as f:
        for intermediate_result in intermediate_results:
            f.write(json.dumps(intermediate_result, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved {len(intermediate_results)} intermediate records")
    
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
            'original_candidate_count': result['original_candidate_count'],
            'num_groups': result['num_groups'],
            'final_candidate_count': result['final_candidate_count']
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"   ✓ Saved {len(csv_data)} records")

    # 9. Print summary
    print("\n" + "=" * 70)
    print("Global summary")
    print("=" * 70)
    print(f"Queries: {summary['num_queries']}")
    print(f"Average final candidate count: {summary['avg_final_candidate_count']:.1f}")
    print(f"Average unique selected per query: {summary['avg_unique_selected']:.1f}")
    
    print("\n" + "=" * 70)
    print("✓ Stage 2 completed")
    print("=" * 70)


if __name__ == "__main__":
    main()

