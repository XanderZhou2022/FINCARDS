#!/usr/bin/env python3
"""
Stage 1: Parallel BM25 candidate generation
- Run intra-document BM25 retrieval for every question
- Use dynamic N = clamp(ceil(r * L), min_N=50, max_N=100)
- Emit detailed per-query records and a global summary
"""

import json
import math
import re
from collections import Counter
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

# Paths
QUESTIONS_FILE = "../data/finbenchqa/filtered_data/questions.json"
UNIQUE_CHUNKS_FILE = "../data/finbenchqa/filtered_data/unique_chunks.json"
OUTPUT_DIR = Path(".")
OUTPUT_DETAILED = OUTPUT_DIR / "stage1_candidates_detailed.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "stage1_summary.json"

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Dynamic N parameters
CANDIDATE_RATIO_R = 0.5  # default candidate ratio (50%)
MIN_N = 60
MAX_N = 150  # clipped upper bound for the dynamic candidate size

# Parallelism
MAX_WORKERS = 256  # number of worker threads


def tokenize(text: str) -> List[str]:
    """Simple word-level tokenization."""
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


class BM25:
    """Minimal BM25 implementation for intra-document ranking."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.doc_freqs: List[Counter] = []
        self.idf = {}
        self.avgdl = 0.0

    def fit(self, corpus: List[str]):
        """Fit BM25 statistics on the provided corpus."""
        self.corpus = corpus
        tokenized_corpus = [tokenize(doc) for doc in corpus]

        # Document frequency
        df = Counter()
        for doc in tokenized_corpus:
            df.update(set(doc))

        # IDF
        num_docs = len(corpus)
        for term, freq in df.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        # Average document length
        doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0

        # Term frequencies per document
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]

    def get_score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score between a query and a document."""
        q_tokens = tokenize(query)
        doc_freq = self.doc_freqs[doc_idx]
        dl = sum(doc_freq.values())

        score = 0.0
        for t in q_tokens:
            if t in doc_freq:
                idf = self.idf.get(t, 0.0)
                tf = doc_freq[t]
                num = idf * tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += num / den
        return score

    def get_scores(self, query: str) -> List[float]:
        """Compute BM25 scores for a query against all documents."""
        return [self.get_score(query, i) for i in range(len(self.corpus))]


def compute_dynamic_N(L: int, r: float = CANDIDATE_RATIO_R, min_N: int = MIN_N, max_N: int = MAX_N) -> int:
    """Compute dynamic candidate size N = clamp(ceil(r * L), min_N, max_N)."""
    N = math.ceil(r * L)
    return max(min_N, min(N, max_N))


def process_single_question(
    question_id: str,
    question_info: Dict,
    unique_chunks_data: Dict,
    r: float = CANDIDATE_RATIO_R
) -> Dict:
    """Run BM25 retrieval for a single question."""
    try:
        doc_id = question_info['unique_chunks_id']
        question_text = question_info.get('question_text', '').strip()
        
        if not question_text:
            return {
                'error': 'Missing question_text',
                'question_id': question_id,
                'doc_id': doc_id
            }
        
        # Gather all chunks for the document
        if doc_id not in unique_chunks_data:
            return {
                'error': f'Document {doc_id} not found',
                'question_id': question_id,
                'doc_id': doc_id
            }
        
        doc_info = unique_chunks_data[doc_id]
        chunks = doc_info['chunks']
        chunk_ids = [c['uid'] for c in chunks]
        chunk_texts = [c['text'] for c in chunks]
        
        L = len(chunks)
        N = compute_dynamic_N(L, r)
        
        # BM25 retrieval
        bm25 = BM25(k1=BM25_K1, b=BM25_B)
        bm25.fit(chunk_texts)
        scores = bm25.get_scores(question_text)
        
        # Rank chunks by BM25 score
        scored = list(zip(chunk_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Build quick score map
        score_map = {cid: s for cid, s in scored}
        rank_map = {cid: i + 1 for i, (cid, _) in enumerate(scored)}
        
        # Top-N candidates
        top_n = scored[:N]
        candidates = [
            {
                'chunk_id': cid,
                'bm25_score': round(s, 4),
                'bm25_rank_in_doc': rank_map[cid]
            }
            for cid, s in top_n
        ]
        
        # Build result payload (gold information intentionally omitted)
        result = {
            'query_id': question_id,
            'doc_id': doc_id,
            'question_text': question_text,
            'L_total_chunks': L,
            'candidate_ratio_r': r,
            'candidate_topN': N,
            'candidates': candidates
        }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'question_id': question_id,
            'doc_id': question_info.get('unique_chunks_id', 'unknown')
        }


def compute_summary(all_results: List[Dict]) -> Dict:
    """Compute global summary statistics (gold-free)."""
    valid_results = [r for r in all_results if 'error' not in r]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    num_queries = len(valid_results)
    
    # Document length distribution
    L_values = [r['L_total_chunks'] for r in valid_results]
    avg_L = sum(L_values) / len(L_values) if L_values else 0
    median_L = sorted(L_values)[len(L_values) // 2] if L_values else 0
    
    N_values = [r['candidate_topN'] for r in valid_results]
    N_distribution = Counter(N_values)
    
    summary = {
        'num_queries': num_queries,
        'avg_L_total_chunks': round(avg_L, 2),
        'median_L_total_chunks': median_L,
        'N_distribution': dict(N_distribution)
    }
    
    return summary


def main():
    print("=" * 60)
    print("Stage 1: Parallel BM25 candidate generation")
    print("=" * 60)
    print("Configuration:")
    print(f"  BM25 k1={BM25_K1}, b={BM25_B}")
    print(f"  Candidate ratio r={CANDIDATE_RATIO_R}, min_N={MIN_N}, max_N={MAX_N}")
    print(f"  Worker threads: {MAX_WORKERS}")
    print()
    
    # Read input data
    print("Loading data files...")
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    with open(UNIQUE_CHUNKS_FILE, 'r', encoding='utf-8') as f:
        unique_chunks_data = json.load(f)
    
    print(f"  Questions: {len(questions)}")
    print(f"  Documents: {len(unique_chunks_data)}")
    print()
    
    # Parallel processing
    print(f"Starting parallel processing ({MAX_WORKERS} threads)...")
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_question,
                qid, qinfo, unique_chunks_data, CANDIDATE_RATIO_R
            ): qid
            for qid, qinfo in questions.items()
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            all_results.append(result)
            
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {completed}/{len(questions)} ({completed/len(questions)*100:.1f}%), "
                      f"elapsed: {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"\nProcessing complete! Elapsed: {elapsed:.1f}s")
    print()
    
    # Save detailed results
    print(f"Saving detailed results to {OUTPUT_DETAILED}...")
    with open(OUTPUT_DETAILED, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"  ✓ Saved {len(all_results)} records")
    
    # Compute and save summary
    print("\nComputing global summary...")
    summary = compute_summary(all_results)
    
    print(f"Saving summary to {OUTPUT_SUMMARY}...")
    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("  ✓ Saved")

    # Print summary
    print("\n" + "=" * 60)
    print("Global summary")
    print("=" * 60)
    print(f"Queries: {summary['num_queries']}")
    print(f"Average document length: {summary['avg_L_total_chunks']:.1f} chunks")
    print(f"Median document length: {summary['median_L_total_chunks']} chunks")
    print(f"Candidate count distribution (Top-N): {summary['N_distribution']}")
    
    print("\n" + "=" * 60)
    print("✓ Stage 1 completed")
    print("=" * 60)


if __name__ == "__main__":
    main()

