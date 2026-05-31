#!/usr/bin/env python3
"""
Stage 0 (Intent Mapping): Convert raw questions into structured intents for FINCARDS.
- Loads questions.json
- Calls LLM with a strict JSON-only prompt
- Writes intents to JSONL for downstream stages
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

# API configuration (environment-provided; no hardcoded secrets)
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required.")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# Paths and processing settings
QUESTIONS_FILE = "../data/finbenchqa/filtered_data/questions.json"
OUTPUT_FILE = "stage0_query_intents.jsonl"
MAX_WORKERS = 8
MAX_RETRIES = 1
RETRY_DELAY = 2  # seconds


INTENT_PROMPT = """You are a financial QA intent extractor for SEC filings.
Given a user question, produce a STRICT JSON object capturing the structured intent.

QUESTION:
{question_text}

RETURN JSON ONLY with the following fields:
{{
  "topic": "one of [Revenue, Costs/Expenses, Profitability, Liquidity, Guidance/Outlook, Risk, ESG, Market, AccountingPolicy, Legal, MD&A, FinancialStatements, Other]",
  "entities": ["company or segment names if explicitly mentioned, else []"],
  "metrics": ["financial metrics (e.g., revenue, cogs, ebit, eps, margin, fcf, capex, debt, interest, liquidity, guidance)"],
  "temporal_scope": {{
    "type": "explicit | implicit | none",
    "periods": ["normalized periods mentioned (e.g., FY2023, Q1 2024) if any, else []"],
    "granularity": "year | quarter | month | mixed | none"
  }},
  "requires_numeric_evidence": true/false,
  "relation": "lookup | trend | comparison | explanation | definition | policy",
  "keywords": ["up to 8 salient keywords/phrases from the question"]
}}

RULES:
- Return JSON only, no prose, no markdown.
- If uncertain about a field, set to "Other"/"none"/[] as appropriate, do NOT hallucinate.
- Temporal normalization: prefer fiscal expressions (FY2023, Q4 2022, latest quarter).
- requires_numeric_evidence: true if the question expects numbers (values, deltas, ratios).
- relation:
  * lookup: asks for a value
  * trend: asks how it changed over time
  * comparison: asks versus another period/entity
  * explanation: asks for drivers/causes/impact
  * definition: asks what/meaning
  * policy: asks about accounting/policy/method
"""


def build_prompt(question_text: str) -> str:
    """Insert the question into the intent prompt."""
    return INTENT_PROMPT.format(question_text=question_text.strip())


def parse_json_from_response(response_text: str) -> Dict:
    """Extract JSON object from the LLM response."""
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in response")
    return json.loads(json_match.group(0))


def generate_intent(question_id: str, question_text: str, retry: int = 0) -> Dict:
    """Generate structured intent for a single question."""
    try:
        prompt = build_prompt(question_text)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        response_text = completion.choices[0].message.content
        intent = parse_json_from_response(response_text)
        return {"question_id": question_id, "intent": intent, "status": "success"}
    except Exception as e:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (retry + 1))
            return generate_intent(question_id, question_text, retry + 1)
        return {
            "question_id": question_id,
            "status": "error",
            "error": str(e),
        }


def load_questions() -> Dict[str, Dict]:
    """Load questions.json."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: List[Dict], path: Path):
    """Save records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("Stage 0: Query Intent Mapping")
    print("=" * 60)

    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {QUESTIONS_FILE}")

    tasks = [(qid, qinfo.get("question_text", "")) for qid, qinfo in questions.items()]

    results: List[Dict] = []
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_intent, qid, qtext): qid
            for qid, qtext in tasks
        }

        for idx, future in enumerate(as_completed(futures), 1):
            qid = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result.get("status") == "success":
                    success += 1
                    print(f"[{idx}/{len(tasks)}] ✓ {qid}")
                else:
                    failed += 1
                    print(f"[{idx}/{len(tasks)}] ✗ {qid}: {result.get('error')}")
            except Exception as e:
                failed += 1
                results.append({"question_id": qid, "status": "error", "error": str(e)})
                print(f"[{idx}/{len(tasks)}] ✗ {qid}: {str(e)}")

    save_jsonl(results, Path(OUTPUT_FILE))

    print("\nCompleted intent mapping.")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

