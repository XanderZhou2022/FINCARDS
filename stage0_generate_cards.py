#!/usr/bin/env python3
"""
Batch generate Chunk Cards with concurrent processing.
Processes chunks with a small worker pool and saves progress/results incrementally.
"""

import json
import re
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

# API configuration (environment-provided; no hardcoded secrets)
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# Paths and processing settings
INPUT_FILE = "filtered_data/unique_chunks.json"
OUTPUT_FILE = "filtered_data/chunk_cards.json"
PROGRESS_FILE = "filtered_data/card_generation_progress.json"
MAX_WORKERS = 5  # number of concurrent chunk tasks
MAX_RETRIES = 1  # max retry count
RETRY_DELAY = 2  # retry delay in seconds


def create_chunk_card_prompt(chunk_uid, chunk_text):
    """
    Build the prompt for generating a Chunk Card with Semantic Sketch.
    """
    prompt = f"""You are a financial document analysis expert implementing FINCARDS, a research framework for structured evidence alignment interfaces.

Your task is to generate a complete Chunk Card with Semantic Sketch for a text chunk extracted from a financial document (10-K filing). 

IMPORTANT CONCEPTS:
- Chunk Card: A structured representation of "under what conditions can this chunk serve as evidence, and what are its capability boundaries." It is NOT a feature extraction task.
- Semantic Sketch: A controlled, low-bandwidth semantic proxy for Stage II. It provides "what this chunk is about" without full text, reasoning chains, or QA-style answers.

CHUNK INFORMATION:
- Chunk UID: {chunk_uid}
- Chunk Text: 
{chunk_text}

REQUIREMENTS FOR COMPLETE OUTPUT:

You must generate a complete JSON object with the following 9 sections (8 Chunk Card sections + 1 Semantic Sketch):

1. Identity & Structure (for location and audit purposes):
{{
  "chunk_id": "{chunk_uid}",
  "doc_id": "extracted_from_context",
  "chunk_text": "...",  // full chunk text
  "section_path": [...],  // inferred section hierarchy if possible, else []
  "chunk_index": <number>  // position in document if inferable
}}

2. Evidence Role (MUST be explicit):
{{
  "claim_role": "primary_evidence | supporting_context | definition | caveat | boilerplate | structural_heading"
}}

CRITICAL: If the chunk is purely structural (e.g., "PART I", "Item 1", "BALANCE SHEETS", table headers only), it MUST be "structural_heading".

3. Evidence Type:
{{
  "evidence_type": "table_numeric | narrative_numeric | qualitative_explanation | policy_text | guidance | none"
}}

CRITICAL: "none" is ONLY allowed for structural_heading. For structural_heading, evidence_type MUST be "none".

4. Answerability Profile (capability distribution, NOT labels):
{{
  "answerability_profile": {{
    "single_fact": 0.0-1.0,  // can answer single fact questions
    "comparison": 0.0-1.0,   // can answer comparison questions
    "trend": 0.0-1.0,        // can answer trend questions
    "aggregation": 0.0-1.0,  // can answer aggregation questions
    "attribution": 0.0-1.0   // can answer attribution questions
  }}
}}

CRITICAL: For structural_heading, ALL values MUST be 0.0. This is a capability distribution for reranking alignment, not content matching.

5. Temporal Anchor:
{{
  "temporal_anchor_quality": "explicit | implicit | none",
  "temporal_span": {{
    "start": "YYYY-MM | null",
    "end": "YYYY-MM | null",
    "granularity": "year | quarter | month | none"
  }}
}}

CRITICAL: 
- Date format MUST be "YYYY-MM" (e.g., "2022-01", "2023-09") or null
- If no temporal information exists, MUST explicitly set to "none"
- This is key for temporal mismatch and stability analysis

6. Scope & Measurement:
{{
  "scope_signature": {{
    "entity_scope": "subsidiary | segment | consolidated | unknown",
    "geography": [...],  // list of geographic regions if mentioned
    "product": [...]     // list of products/segments if mentioned
  }},
  "measurement_basis_signature": {{
    "basis": "GAAP | non-GAAP | adjusted | reported | unknown",
    "definition_present": true | false
  }}
}}

CRITICAL: "unknown" is required if uncertain, NOT omitted. This prevents scope mismatch and numeric drift.

7. Verifiability & Evidence Strength (REPLACED with detailed structure):
{{
  "verifiability": {{
    "has_numeric_claim": true | false,
    "numeric_units": ["USD", "%"],  // array of units, can be empty []
    "numeric_role": ["value", "delta", "ratio", "pct_change"],  // can be multiple
    "comparison_type": "yoy | qoq | period_vs_period | none",
    "table_signature": {{
      "present": true | false,
      "table_role": "value_table | comparison_table | breakdown_table | reconciliation_table | time_series_table | unknown",
      "row_axis": "time | entity | product | geography | metric | mixed | unknown",
      "column_axis": "time | metric | entity | comparison | mixed | unknown",
      "primary_comparison_axis": "time | entity | metric | none",
      "time_columns_present": true | false,
      "comparison_columns_present": true | false,
      "contains_delta": true | false,
      "contains_ratio": false,
      "contains_pct_change": true | false,
      "has_total_row_or_col": true | false,
      "measurement_basis_cues": ["GAAP", "non-GAAP", "adjusted", "reported"],  // can be empty []
      "table_span": {{
        "start": "YYYY | null",
        "end": "YYYY | null",
        "granularity": "year | quarter | month | none"
      }}
    }},
    "text_comparison_cues": {{
      "has_compared_with_phrase": true | false,
      "has_primary_driver_phrase": true | false,
      "has_offset_phrase": true | false
    }}
  }},
  "evidence_strength_score": 0.0-1.0
}}

CRITICAL RULES for Verifiability:
- has_numeric_claim: true if chunk contains numeric claims with financial/metric terms (revenue, margin, EPS, income); false if only risk language without numbers
- numeric_units: Extract from text/headers ($, "million", "%", etc.); empty [] if no units found (do NOT guess)
- numeric_role: Can be multiple values - "value" (absolute), "delta" (change amount), "ratio" (ratio/ratio), "pct_change" (percentage change)
- comparison_type: "yoy" (year-over-year), "qoq" (quarter-over-quarter), "period_vs_period" (any two periods), or "none"
- table_signature.present: true if table structure exists, false otherwise
- If table_signature.present=false, other table_signature fields can be omitted or minimal
- If table_signature.present=true, ALL table_signature fields MUST have values (use "unknown" if uncertain)
- table_role: Determined by table function - "comparison_table" if has Change/% Change columns, "time_series_table" if ≥3 periods, "breakdown_table" if rows are products/regions/segments, "reconciliation_table" if GAAP/non-GAAP reconciliation
- row_axis/column_axis: Semantic meaning of axes, not field names - "time" if years/quarters, "entity" if business units, "product" if products, "geography" if regions, "mixed" if multiple types, "unknown" if uncertain
- primary_comparison_axis: Which axis comparison occurs along - "time" if columns are 2023/2022/Change, "entity" if comparing different entities, "metric" if comparing different metrics, "none" if no comparison
- text_comparison_cues: Text-level comparison/causal signals - "compared with"/"versus"/"vs." → has_compared_with_phrase, "primarily due to"/"mainly driven by" → has_primary_driver_phrase, "partially offset by"/"offset by" → has_offset_phrase

SPECIAL CASE for structural_heading:
- has_numeric_claim: MUST be false
- table_signature.present: MUST be false
- numeric_units: MUST be []
- numeric_role: MUST be []
- comparison_type: MUST be "none"

8. Risk & Limitation Signals:
{{
  "boilerplate_likelihood": 0.0-1.0,
  "redundancy_cluster_id": null,  // can be null
  "cluster_density": 0,  // can be 0
  "limitations": [
    // possible values: "no_numbers", "no_period", "forward_looking_only", 
    // "partial_scope", "definition_only", "no_content"
    // empty list if no limitations
  ]
}}

This is soft constraints for error taxonomy and instability attribution, NOT filtering rules.

9. Semantic Sketch (MANDATORY - Stage II semantic proxy):
{{
  "semantic_sketch": {{
    "claim_summary": "One-sentence factual description of the main claim.",
    "topic_anchors": [
      "key concept 1",
      "key concept 2",
      "key concept 3"
    ],
    "numeric_placeholders": [
      "YoY increase",
      "$X million"
    ]
  }}
}}

CRITICAL CONSTRAINTS for Semantic Sketch:
- claim_summary: 
  * ≤ 30 tokens
  * Single sentence, declarative statement
  * NO copying from original text
  * Provides "what this chunk is about" without full text
- topic_anchors:
  * 3-6 items
  * Concept-level, NOT entity list
  * Key concepts that anchor the semantic meaning
- numeric_placeholders:
  * Optional (can be empty array [])
  * Do NOT fill in actual numeric values
  * Use placeholders like "YoY increase", "$X million", "percentage change"

SPECIAL CASE - Structural Headings:
For chunks like "PART I", "Item 1", "BALANCE SHEETS" (purely structural with no actual content):
- claim_role: "structural_heading"
- evidence_type: "none"
- answerability_profile: ALL 0.0
- temporal_anchor_quality: "none"
- verifiability:
  {{
    "has_numeric_claim": false,
    "numeric_units": [],
    "numeric_role": [],
    "comparison_type": "none",
    "table_signature": {{ "present": false }},
    "text_comparison_cues": {{
      "has_compared_with_phrase": false,
      "has_primary_driver_phrase": false,
      "has_offset_phrase": false
    }}
  }}
- evidence_strength_score: 0.0
- limitations: ["no_content"]
- semantic_sketch:
  {{
    "claim_summary": "Section heading without substantive content.",
    "topic_anchors": ["section title"],
    "numeric_placeholders": []
  }}

This is NOT "ignoring noise" but explicitly declaring the chunk has NO evidence capability.

OUTPUT FORMAT:
Return ONLY a valid JSON object with all 9 sections (8 Chunk Card sections + 1 Semantic Sketch). Do not include any explanatory text, markdown formatting, or code blocks. The output must be parseable JSON.

VALIDATION CHECKLIST:
Before outputting, verify:
1. All 9 sections are present (8 Chunk Card + 1 Semantic Sketch)
2. For structural_heading: evidence_type is "none", all answerability_profile values are 0.0, verifiability.has_numeric_claim=false, table_signature.present=false
3. Temporal information: if no dates found, set to "none"; date format is "YYYY-MM" or null
4. Scope: use "unknown" if uncertain, not omitted
5. All boolean fields are true/false (not strings)
6. All numeric scores are between 0.0 and 1.0
7. Semantic Sketch: claim_summary ≤ 30 tokens, topic_anchors 3-6 items, no actual numbers in placeholders
8. Verifiability: If table_signature.present=true, ALL table_signature fields must have values (use "unknown" if uncertain)
9. Verifiability: numeric_units and numeric_role are arrays (can be empty [])
10. JSON is valid and parseable

Now generate the complete Chunk Card with Semantic Sketch for the provided chunk:"""

    return prompt


def generate_chunk_card(chunk_uid, chunk_text, retry_count=0):
    """
    Generate a Card for a single chunk.
    """
    try:
        prompt = create_chunk_card_prompt(chunk_uid, chunk_text)
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
        )
        
        response_text = completion.choices[0].message.content
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            chunk_card = json.loads(json_str)
            return {
                'chunk_uid': chunk_uid,
                'card': chunk_card,
                'status': 'success'
            }
        else:
            return {
                'chunk_uid': chunk_uid,
                'status': 'error',
                'error': 'JSON extraction failed',
                'raw_response': response_text[:500]
            }
            
    except json.JSONDecodeError as e:
        return {
            'chunk_uid': chunk_uid,
            'status': 'error',
            'error': f'JSON decode error: {str(e)}',
            'raw_response': response_text[:500] if 'response_text' in locals() else 'N/A'
        }
    except Exception as e:
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (retry_count + 1))
            return generate_chunk_card(chunk_uid, chunk_text, retry_count + 1)
        else:
            return {
                'chunk_uid': chunk_uid,
                'status': 'error',
                'error': str(e)
            }


def load_progress():
    """Load progress file if present."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'processed_chunks': set(),
        'results': {}
    }


def save_progress(progress):
    """Persist progress to disk."""
    # Convert set to list for JSON serialization
    progress_to_save = {
        'processed_chunks': list(progress['processed_chunks']),
        'results': progress['results']
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_to_save, f, indent=2, ensure_ascii=False)


def load_results():
    """Load existing results if any."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_results(results):
    """Save results to disk."""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("Batch generate Chunk Cards")
    print("=" * 60)
    
    # Load input data
    print(f"\nReading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        unique_chunks_data = json.load(f)
    
    # Load progress
    print("Loading progress file...")
    progress = load_progress()
    processed_chunks = set(progress.get('processed_chunks', []))
    
    # Load existing results
    results = load_results()
    
    # Gather chunks needing processing
    all_chunks = []
    for combo_id, combo_info in unique_chunks_data.items():
        for chunk in combo_info['chunks']:
            chunk_uid = chunk['uid']
            if chunk_uid not in processed_chunks:
                all_chunks.append({
                    'chunk_uid': chunk_uid,
                    'chunk_text': chunk['text'],
                    'combo_id': combo_id
                })
    
    total_chunks = len(all_chunks)
    print("\nCorpus statistics:")
    print(f"  Total chunks: {sum(info['chunk_count'] for info in unique_chunks_data.values()):,}")
    print(f"  Processed: {len(processed_chunks):,}")
    print(f"  Pending: {total_chunks:,}")
    
    if total_chunks == 0:
        print("\nAll chunks are already processed.")
        return
    
    print(f"\nStarting batch processing (workers: {MAX_WORKERS})...")
    print(f"Expected to process {total_chunks} chunks\n")
    
    # Concurrent processing
    completed = 0
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        future_to_chunk = {
            executor.submit(generate_chunk_card, chunk['chunk_uid'], chunk['chunk_text']): chunk
            for chunk in all_chunks
        }
        
        # Handle completed tasks
        for future in as_completed(future_to_chunk):
            chunk_info = future_to_chunk[future]
            chunk_uid = chunk_info['chunk_uid']
            
            try:
                result = future.result()
                completed += 1
                
                if result['status'] == 'success':
                    successful += 1
                    results[chunk_uid] = result['card']
                    processed_chunks.add(chunk_uid)
                    print(f"[{completed}/{total_chunks}] ✓ {chunk_uid}")
                else:
                    failed += 1
                    results[chunk_uid] = {
                        'error': result.get('error', 'Unknown error'),
                        'raw_response': result.get('raw_response', 'N/A')
                    }
                    processed_chunks.add(chunk_uid)  # mark processed even on failure
                    print(f"[{completed}/{total_chunks}] ✗ {chunk_uid}: {result.get('error', 'Unknown error')}")
                
                # Save every 10 chunks
                if completed % 10 == 0:
                    progress['processed_chunks'] = processed_chunks
                    progress['results'] = results
                    save_progress(progress)
                    save_results(results)
                    print(f"  Progress saved (success: {successful}, failed: {failed})")
                    
            except Exception as e:
                completed += 1
                failed += 1
                results[chunk_uid] = {'error': str(e)}
                processed_chunks.add(chunk_uid)
                print(f"[{completed}/{total_chunks}] ✗ {chunk_uid}: Exception - {str(e)}")
    
    # Final save
    print(f"\nProcessing finished!")
    print(f"  Total: {completed}/{total_chunks}")
    print(f"  Success: {successful}")
    print(f"  Failed: {failed}")
    
    progress['processed_chunks'] = processed_chunks
    progress['results'] = results
    save_progress(progress)
    save_results(results)
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Progress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()

