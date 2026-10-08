# LLM-Driven PDPL Disclosure Evaluation

Evaluate privacy-policy disclosures against 15 rubric criteria using ALLaM-7B, LLaMA-3.1-8B, Qwen-2.5-7B and Mistral-7B. Predictions indicate disclosed (1) or missing, unclear or insufficient disclosure (0).

Four strategies are evaluated over five repetitions: `section_rag`, `full_rag`, `section_norag` and `full_norag`. The intended dataset contains 21 applications; execution follows `configs/applications.yaml`.

## Files

| Path | Contents |
|---|---|
| `configs/` | Applications, models and experiment settings |
| `data/policies/` | Processed policy DOCX files |
| `data/pdpl/IRPDPL.docx` | Law retrieval corpus |
| `data/ground_truth/PDPL_Results.xlsx` | Expert labels: `Expert Analysis` sheet |
| `rubric/` | The 15 criteria |
| `prompts/` | Four strategy templates |
| `src/pdpl_eval/` | Pipeline implementation |
| `scripts/` | Execution, checks and evaluation |

## Setup

Use Python 3.11 or 3.12 and install a GPU-compatible PyTorch build. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_experiments.py --check-inputs
```

Set `HF_TOKEN` using an account with access to the configured models. Keep the actual token private; `.env.example` contains a placeholder.

Windows Command Prompt:

```bat
set "HF_TOKEN=YOUR_PRIVATE_TOKEN"
```

Linux shell:

```bash
export HF_TOKEN="YOUR_PRIVATE_TOKEN"
```

Freeze model revisions and inspect the execution plan:

```bash
python scripts/run_experiments.py --pin-revisions
python scripts/run_experiments.py --dry-run
```

## Preliminary checks

Check full-policy RAG, then run one application across all four models and strategies for one repetition:

```bash
python scripts/run_full_policy_rag_test.py --app circlys --model allam-7b
python scripts/run_smoke_test.py --app circlys
```

The pilot saves 16 run records and stops for review. It does not start the full experiment.

## Full experiment

After reviewing the preliminary results:

```bash
python scripts/run_experiments.py --resume
python scripts/validate_results.py
python scripts/consolidate_results.py --evaluate
```

Matching successful preliminary records are reused. Models load sequentially and unload before the next model; downloaded weights remain cached.

## Results

| Directory | Outputs |
|---|---|
| `results/raw/` | Predictions, prompts, responses, section vectors and run metadata |
| `results/logs/` | Logs and completion reports |
| `results/preflight/` | RAG context checks |
| `results/archive/` | Replaced records |
| `results/consolidated/` | Combined CSV/JSONL results |
| `results/analysis/` | Evaluation tables and false-positive heatmap |

Evaluation reports dataset-wide and criterion-level metrics, including means and standard deviations across complete repetitions. OR is the primary section aggregation; AND, strict majority and at least two are compared using saved section predictions.

Incomplete predictions are recorded as `parse_failed`, without zero padding. Preserve the complete results directory, inputs, pinned configurations and code version. Additional plots can be generated later without rerunning the models.
