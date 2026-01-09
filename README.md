## FINCARDS Retrieval Pipeline

This repository contains the implementation of the **FINCARDS** retrieval pipeline used in our paper.  
The pipeline operates in **four stages**:

- **Stage 0a (`stage0_generate_cards.py`)**:  
  - Generates chunk Cards (with Semantic Sketch) from `unique_chunks.json` using an API model.  
  - Inputs:  
    - `../data/finbenchqa/filtered_data/unique_chunks.json`  
  - Outputs:  
    - `filtered_data/chunk_cards.json`  
    - `filtered_data/card_generation_progress.json`

- **Stage 0b (`stage0_query_intent.py`)**:  
  - Maps each question to a structured intent JSON for downstream alignment.  
  - Inputs:  
    - `../data/finbenchqa/filtered_data/questions.json`  
  - Outputs:  
    - `stage0_query_intents.jsonl`

- **Stage 1 (`stage1_lexical_bm25.py`)**:  
  - Parallel BM25 retrieval over intra-document chunks for every question.  
  - Uses dynamic candidate size \(N = \lceil r \cdot L \rceil\) with clamping between `MIN_N` and `MAX_N`.  
  - Inputs:  
    - `../data/finbenchqa/filtered_data/questions.json`  
    - `../data/finbenchqa/filtered_data/unique_chunks.json`  
  - Outputs (in `stage1` folder or current directory, depending on where you run it):  
    - `stage1_candidates_detailed.jsonl`  
    - `stage1_summary.json`

- **Stage 2 (`stage2_card_rerank.py`)**:  
  - Agent-based grouping, filtering, and reranking over Stage 1 candidates using FINCARDS.  
  - Requires OpenAI-compatible API for the card-based agent.  
  - Inputs:  
    - `../stage1/stage1_candidates_detailed.jsonl`  
    - `../data/finbenchqa/filtered_data/` (chunk card JSON files from Stage 0a)  
  - Outputs (in `stage2` directory):  
    - `stage2_results_detailed.jsonl`  
    - `stage2_intermediate_results.jsonl`  
    - `stage2_summary.json`  
    - `stage2_results.csv`

- **Stage 3 (`stage3_bootstrap_listwise.py`)**:  
  - Bootstrap-based listwise ranking and aggregation over Stage 2 candidates.  
  - Runs multiple rounds of group-wise listwise ranking with Borda-style aggregation.  
  - Inputs:  
    - `../stage2/stage2_results_detailed.jsonl`  
    - `../data/finbenchqa/filtered_data/` (chunk card JSON files from Stage 0a)  
  - Outputs (in `stage3` directory):  
    - `stage3_results_detailed.jsonl`  
    - `stage3_intermediate_results.jsonl`  
    - `stage3_summary.json`  
    - `stage3_results.csv`

### Environment & Dependencies

- **Python**: 3.9+ recommended  
- **Core libraries** (non-exhaustive, inferred from scripts):  
  - `pandas`  
  - `numpy`  
  - `openai` (or compatible client library providing `OpenAI`)  

You can install them via:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, a minimal starting point is:

```bash
pip install pandas numpy openai
```

### API Configuration

Stages 2 and 3 require an API key and model name:

- **Environment variables**:
  - `OPENAI_API_KEY` – your API key (required for Stage 2 and Stage 3).  
  - `OPENAI_MODEL_NAME` – model name (optional, defaults are set in the scripts).

Example:

```bash
export OPENAI_API_KEY="sk-xxxxx"
export OPENAI_MODEL_NAME="gpt-5-mini-2025-08-07"
```

### Data Layout

The scripts expect the **FinBenchQA**-style data layout:

- `../data/finbenchqa/filtered_data/questions.json`  
- `../data/finbenchqa/filtered_data/unique_chunks.json`  
- Additional chunk card JSON files under `../data/finbenchqa/filtered_data/` used in Stages 2 and 3.

Make sure you run each stage from its intended directory (e.g., `stage1/`, `stage2/`, `stage3/`) so that the relative paths in the scripts resolve correctly.

### Running the Pipeline

1. **Stage 1 – BM25 candidates**

```bash
cd stage1
python stage1_lexical_bm25.py
```

2. **Stage 2 – Card-based reranking**

```bash
cd ../stage2
python stage2_card_rerank.py
```

3. **Stage 3 – Bootstrap listwise ranking**

```bash
cd ../stage3
python stage3_bootstrap_listwise.py
```

Each stage reads the outputs from the previous stage and writes its own detailed and summary artifacts.

### Notes

- The scripts are designed to be **deterministic in Stage 1**, and **stochastic in Stage 3** (bootstrap + random grouping), so you may see slight variance between runs in Stage 3.  
- Hyperparameters such as candidate ratios, group sizes, and top‑K can be adjusted directly in the corresponding Python files.


