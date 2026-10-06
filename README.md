# PDPL disclosure evaluation pipeline

Reproducible, resumable pipeline that asks local open-weight LLMs whether privacy policies of Saudi fintech applications **disclose** the
17 observable, PDPL-aligned items of a rubric (data-subject rights A1–A5, processing principles B1–B7, policy clauses C1–C3, operational
obligations D1–D2). It was refactored from the notebook `notebooks/PDPL_Evaluate_with_Arabic.ipynb`.

> **Scope.** The task evaluates whether a policy's text *contains observable PDPL-aligned disclosures*. A `1` means the model judged the
> requirement to be explicitly addressed in the policy text; a `0` means missing, unclear, or insufficient. This is **not** a determination of
> legal compliance, and the outputs of 7–8B models must not be used as a compliance verdict.

## Strategies

Every application × model is run under four strategies (ids are fixed and used in file names and records):

| id | Description | Notebook origin |
|---|---|---|
| `section_rag` | Each policy section is evaluated separately; the 3 most similar PDPL chunks (multilingual-e5-small + FAISS, query = section text) are added to the prompt; per-section 0/1 vectors are combined with **OR** | baseline "Section-by-Section (OR)" |
| `full_rag` | Whole policy in one prompt plus the top-3 PDPL chunks retrieved with the rubric text as query | Experiment 1 "Full-Policy+Law" |
| `section_norag` | As `section_rag` but without the PDPL excerpts | **new** |
| `full_norag` | Whole policy in one prompt, no PDPL text | baseline "Full-Policy" |

Each (application, model, strategy) is run **5 times** (`runs: 5`). With the shipped configuration: 19 apps × 4 models × 4 strategies × 5 runs = **1,520 runs**.
Models (unchanged): ALLaM-7B, LLaMA-3.1-8B, Qwen-2.5-7B, Mistral-7B (local GPU, fp16, `device_map=auto`).

Section-based runs store **every section's prompt, retrieval, raw response and 17-value vector**, so OR / AND / majority / threshold aggregation
(`src/pdpl_eval/aggregation.py`) can be recomputed later without re-running any model. The saved final prediction uses OR (the notebook's rule).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.8.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126   # as in the notebook; adapt to your CUDA
pip install -r requirements.txt
cp .env.example .env        # put your Hugging Face token in .env, then:  set -a; source .env; set +a
```
Gated models (Llama-3.1) need accepted licence terms. Optional pre-download: `python scripts/download_models.py`.
A CUDA GPU with enough VRAM for a 7–8B fp16 model (~16 GB+) is needed for real runs. Mock runs need no GPU.

## Data placement
See `data/README.md`. In short: policy DOCX files in `data/policies/`, `IRPDPL.docx` in `data/pdpl/`. Nothing private is shipped and both folders are git-ignored.

## Configuration (`configs/`)
| File | Controls |
|---|---|
| `applications.yaml` | application name, id, policy path (19 apps) |
| `models.yaml` | model ids/labels, HF id, revision, cache dir, dtype, prompt template, optional `context_limit_tokens` |
| `experiment.yaml` | `runs`, enabled strategies and their prompt paths, rubric path, RAG (corpus, embedding model, top-k, separators), generation (`max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`, `do_sample`, `seed_base`), retry/backoff, output dir |
| `demo/` | mock models + synthetic data for `--mock` |

Prompts live in `prompts/`, the rubric in `rubric/rubric_items.txt`. Nothing about applications, paths or models is hard-coded in `src/`.

## Tests (run these first)
```bash
python scripts/run_smoke_test.py                     # 1 app, 1 model, full_norag, 1 run: load → prompt → response → parse → save → validate → resume detection
python scripts/run_full_policy_rag_test.py           # longest policy, 1 model, full_rag, 1 run; prints policy / retrieved-context / prompt / total token lengths
python scripts/run_smoke_test.py --mock              # plumbing check without GPU / data
python scripts/run_full_policy_rag_test.py --mock
python -m pytest -q tests                            # unit + mock end-to-end tests (no GPU needed)
```
Both test scripts write to `results/smoke_test/<timestamp>/` and `results/full_policy_rag_test/<timestamp>/`, never into the main results.
The full-policy+RAG test checks the model's context limit **before loading weights**. If `prompt tokens + max_new_tokens` exceeds the limit it stops
(exit code 3), writes `preflight_report.json` explaining why, and sends nothing. **Content is never truncated.**
Choose another app/model with `--app` / `--model`.

## Full execution
```bash
python scripts/run_experiments.py --dry-run                      # show the plan
python scripts/run_experiments.py --resume                       # run everything (safe to interrupt and re-run)
python scripts/run_experiments.py --app circlys --model qwen-2.5-7b --strategy full_rag --runs 2     # filters (repeat flags to select several)
python scripts/run_experiments.py --strategy section_rag --strategy full_rag --run-numbers 4 5
python scripts/run_experiments.py --force --app circlys --strategy full_norag --run-numbers 3        # redo a completed run (old file archived)
```
Order: for each model (loaded once, released afterwards) → each application → each enabled strategy → each run. After every run the record is written
atomically (temp file + `fsync` + `os.replace`), re-read from disk, and validated (schema, SHA-256 integrity, rubric keys, aggregation consistency) before continuing.

### Resume behaviour
* A run is *complete* only if its file exists, is valid, and has `status: success`. Such runs are skipped (`--resume` is the default behaviour; the flag is accepted for clarity).
* Missing, `failed`, `parse_failed`, or corrupt files are re-run. The previous file is **moved to `results/archive/`**, never silently overwritten or deleted. `--force` does the same for valid runs.
* An interruption (Ctrl-C, crash, power loss) can lose at most the run in progress; saved files are never half-written.
* Temporary errors (OOM, I/O, …) are retried with exponential backoff (`retry:` in config). Non-retryable errors (context limit) and exhausted retries are recorded as `failed` and the pipeline continues.
* Missing policy file / unreachable model / retriever: those runs are *blocked* (not attempted, listed in the report), the rest continue.

## Outputs
```
results/raw/<app_id>/<model_id>/<strategy>/run_01.json   one self-contained record per run
results/archive/...                                        superseded records
results/logs/run_experiments_<ts>.log, events.jsonl        console log + append-only audit trail
results/logs/completion_report_<ts>.{json,md}               completed / failed / parse_failed / invalid / missing (+ skipped & blocked this invocation)
results/consolidated/all_predictions.{jsonl,csv}            long format: one row per run × rubric criterion (successful runs)
results/consolidated/all_section_predictions.{jsonl,csv}    one row per section × criterion × run
results/consolidated/all_runs.csv                           one row per run incl. failures, runtimes, error info
```
Each raw record contains: application, model (id, HF id, requested and resolved revision, dtype), strategy, run, status (`success` / `failed` / `parse_failed`),
error (type, message, attempts), start/end timestamps and runtime, all generation/retrieval/retry parameters, prompt file and SHA-256, rubric items and SHA-256,
policy path and SHA-256, parser rule, aggregation rule, seed(s), raw response(s), parsed prediction(s), retrieval details (retrieved chunks, titles, distances, corpus SHA-256,
embedding model, chunking rule, top-k), token lengths (full-policy strategies), and the environment (Python, platform, GPU, package versions). Example: `docs/sample_results/` (mock data).

```bash
python scripts/validate_results.py            # validates every file + prints completion matrix; exit 1 if invalid/incomplete (--missing-ok to ignore incomplete)
python scripts/consolidate_results.py         # writes results/consolidated/*
```

## Reproducibility and security
* Secrets only through environment variables (`HF_TOKEN`); `.env` is git-ignored; no keys are in the repository. The original notebook had a hard-coded token that was redacted in `notebooks/` — revoke it.
* `.gitignore` excludes policies, PDPL text, ground truth, model weights, environments, caches and generated results.
* Pin `revision:` per model in `configs/models.yaml` for strict reproducibility; resolved hashes are recorded anyway.

## Limitations
* No real-model run was possible while building this repository; only the mock path and notebook-fidelity tests were executed (see `docs/assumptions_and_missing_information.md`).
* Outputs are LLM judgements on disclosure text, not legal findings; the notebook's own analysis found systematic over-estimation.
* Long policies may exceed a model's usable context; such runs are recorded as failed rather than truncated.
* The E5 embedder truncates inputs at 512 tokens (as in the notebook); counts are recorded.
* No multi-process locking; run one writer per results directory.
* Metrics, plots and cross-app statistics are intentionally not part of this repository yet (`src/pdpl_eval/evaluation.py` preserves the notebook's metric code).

More: `docs/notebook_to_repository_mapping.md`, `docs/behavior_changes.md`, `docs/assumptions_and_missing_information.md`.
