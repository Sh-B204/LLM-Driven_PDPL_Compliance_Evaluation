# Notebook → repository mapping

Notebook: `notebooks/PDPL_Evaluate_with_Arabic.ipynb` (143 cells). "Cell" = 0-based index in the notebook JSON.

| Notebook cell(s) | Content | Repository location |
|---|---|---|
| 0–1 | Title, models/approaches description | `README.md` |
| 2 | `pip install ...` | `requirements.txt` (+ torch note) |
| 4 | Imports, GPU report | per-module imports; GPU info in `environment.py` |
| 6 | `HF_TOKEN` (hard-coded), `PDPL_DOCX_PATH`, `GT_XLSX_PATH`, `OUTPUT_DIR`, `rubric_items`, `rubric_text` | token → env var `HF_TOKEN` (`.env.example`); paths → `configs/experiment.yaml`; rubric → `rubric/rubric_items.txt`, `src/pdpl_eval/rubric.py` |
| 8 | `parse_docx_to_texts_metadatas`, `MultilingualE5Embeddings`, FAISS index, `retriever` (k=3) | `policy.py` (`parse_docx_to_texts_metadatas`), `retrieval.py` (`MultilingualE5Embeddings`, `build_retriever`), `top_k_*` in config |
| 8 (first def) | earlier `parse_llm_response` | superseded by the cell-10 definition (same as in notebook, where the later one wins) → `parsing.py` |
| 10 | `parse_llm_response`, `parse_docx_to_sections` | `parsing.py`, `policy.py` |
| 10 | `build_section_query`, `build_full_policy_prompt` | `prompts/section_rag.txt`, `prompts/full_policy.txt`; `prompts.py` |
| 10 | `evaluate`, `load_ground_truth`, `print_result` | `evaluation.py` (preserved, not run by the pipeline); `print_result` dropped (console output only) |
| 12 | `load_hf_model`, `run_plain`, `MODEL_CONFIGS` (4 models), weight pre-download | `models.py` (`HFLocalModel`), `configs/models.yaml`, `scripts/download_models.py` |
| 15, 18, 21, … 69 (19 app cells) | per-app loop: load GT, parse sections, per-model: section-by-section → OR; full policy; unload | `strategies.py` (`run_section_strategy`, `run_full_strategy`), `aggregation.py`, `runner.py`; app list → `configs/applications.yaml` |
| 15 (inner) | section answer parsing: first 0/1 per line, pad/truncate to 17 | `parsing.parse_section_vector` |
| 15 (inner) | full-policy parse + "last 17 values" fallback | `parsing.parse_full_response` |
| 15 (inner) | `np.array(...).max(axis=0)` OR aggregation | `aggregation.aggregate_or` |
| 15 (inner) | `del tok, mdl; gc.collect(); torch.cuda.empty_cache()` | `HFLocalModel.unload`, called in `runner.py` `finally` |
| 16, 19, … 70 (per-app plots) | metrics / heatmaps / confusion matrices | **not ported** (analysis, out of scope); data is preserved for them in `results/` |
| 72–87 | cross-app loading of summary CSVs, plots, ranking, compliance-score error | **not ported** (final statistics are explicitly out of scope) |
| 88 | Final analysis markdown | stays in the notebook |
| 89–91 | Experiment 1 "Law-Grounded Full-Policy": `get_full_pdpl_context`, `build_full_policy_grounded_prompt` | strategy `full_rag`: `prompts/full_policy_rag.txt`, `retrieval.top_k_full`, `strategies.run_full_strategy` |
| 92–107 | Exp. 1 per-app runs, plots, cross-app | per-app runs → strategy `full_rag` (see `behavior_changes.md` §3 for notebook bugs); the Exp. 1 section arm duplicates the baseline `section_rag` |
| 108–123 | Experiment 2 (temperature 0.7, 512 tokens) | **not part of the 4 strategies**; stays in the notebook |
| 124–139 | Experiment 3 (prompt repetition ×2) | **not part of the 4 strategies**; stays in the notebook |
| 140–142 | all-experiment summary plots | not ported |

## Strategy ↔ notebook label

| Strategy id | Notebook label | Origin |
|---|---|---|
| `section_rag` | Section-by-Section (OR) | baseline, cell 15 |
| `full_norag` | Full-Policy | baseline, cell 15 |
| `full_rag` | Full-Policy+Law | Experiment 1, cell 91/93 |
| `section_norag` | — | **new** (no notebook equivalent) |

## Module overview (`src/pdpl_eval/`)

`config.py` (YAML loading, validation, filters) · `rubric.py` · `prompts.py` · `policy.py` (DOCX → sections) · `retrieval.py` (E5 + FAISS) ·
`models.py` (HF local + mock) · `parsing.py` · `aggregation.py` (or/and/majority/threshold) · `strategies.py` (the four strategies) ·
`retry.py` · `storage.py` (atomic save, verify, archive) · `validation.py` · `runner.py` (sequential loop, resume) · `report.py` ·
`consolidate.py` · `preflight.py` (token/context checks) · `environment.py` · `evaluation.py` (preserved notebook metrics) · `demo.py`, `cli.py`, `utils.py`.
