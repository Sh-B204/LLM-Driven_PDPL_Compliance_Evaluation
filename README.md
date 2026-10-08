# LLM-Driven PDPL Disclosure Evaluation

Evaluate processed privacy policies against 15 rubric criteria using ALLaM-7B, LLaMA-3.1-8B, Qwen-2.5-7B and Mistral-7B. A prediction of **1** means the criterion is disclosed; **0** means missing, unclear or insufficient disclosure. These predictions assess policy text and do not establish legal compliance in practice.

The four strategies are `section_rag`, `full_rag`, `section_norag` and `full_norag`. Each uses five repetitions. Section strategies retrieve law excerpts per section when RAG is enabled; full-policy RAG retrieves excerpts using the rubric as the query.

The intended dataset contains **21 applications**; two processed policies remain to be added. Execution uses only the applications listed in `configs/applications.yaml`. With 21 configured applications, the full matrix contains 1,680 run records and 25,200 final criterion predictions. Each application produces 20 records per model, or 80 across four models. Section strategies make multiple generation calls within each record.

## Repository contents

| Path | Purpose |
|---|---|
| `configs/` | Application paths, model settings and experiment settings |
| `data/policies/` | Processed policy DOCX files |
| `data/pdpl/IRPDPL.docx` | Implementing Regulations retrieval corpus |
| `data/ground_truth/PDPL_Results.xlsx` | Expert labels, on the `Expert Analysis` sheet |
| `rubric/rubric_items.txt` | The 15 evaluation criteria |
| `prompts/` | Four strategy templates; `section.txt` is the section prompt without RAG |
| `src/pdpl_eval/` | Policy loading, inference, retrieval, parsing, saving and evaluation |
| `scripts/` | Experiment entry point, pilot checks, validation and consolidation |
| `.env.example` | Access-token placeholder; no real token belongs in Git |

The original notebook and development tests are maintained separately. They are not required to execute this repository.

## Installation and local checks

Run commands from the repository root. Use Python 3.11 or 3.12. Install a PyTorch build suitable for the execution computer, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m compileall -q src scripts
python scripts/run_experiments.py --check-inputs
```

The input check imports the package and validates configured policies, law text, labels and prompt rendering. It does not download model weights or generate predictions. Passing it does not verify GPU execution.

The full dataset requires a GPU capable of running the configured language models and the embedding model. Language models load sequentially: one model processes all selected applications, strategies and repetitions, then unloads before the next. Weights remain cached on disk; they are not all downloaded in advance or automatically deleted.

## Model access and frozen revisions

The GPU operator should use their own Hugging Face account and access token, with any required model access approved. Set `HF_TOKEN` in the process environment. Creating a `.env` file alone does not load it.

Windows Command Prompt:

```bat
set "HF_TOKEN=YOUR_PRIVATE_TOKEN"
```

Linux or Colab shell:

```bash
export HF_TOKEN="YOUR_PRIVATE_TOKEN"
```

Keep the actual token private. Commit only the placeholder in `.env.example`.

Before real execution, freeze the language-model and embedding-model revisions:

```bash
python scripts/run_experiments.py --pin-revisions
python scripts/run_experiments.py --dry-run
```

Pinning retrieves version metadata rather than model weights. Keep the resulting configurations with the saved experiment results.

## Pilot, review and full experiment

First check full-policy RAG on the pilot application:

```bash
python scripts/run_full_policy_rag_test.py --app circlys --model allam-7b
```

This measures prompt length before loading the language-model weights, then executes one `full_rag` run if the request fits. The check does load the embedding model. Requests exceeding the language-model context limit stop without policy truncation.

Then run the pilot:

```bash
python scripts/run_smoke_test.py --app circlys
```

The updated pilot runs **one application, all four configured models, all four strategies and one repetition**: 16 run records. It reuses matching successful results from the earlier RAG check. It saves the results and stops; it never launches the full experiment automatically.

The operator sends the entire `results/` folder for review. Inspect predictions, raw responses and retrieved excerpts against the policy and rubric. A successful structural check does not establish prediction accuracy.

Only after review, start the full matrix separately:

```bash
python scripts/run_experiments.py --resume
python scripts/validate_results.py
python scripts/consolidate_results.py --evaluate
```

Successful pilot records are reused as repetition 1 when their inputs, inference code, settings and frozen revisions match. Use the same results directory. Changes that affect those fingerprints invalidate reuse; complete the code and prompt adjustments before the pilot.

After the temporary pilot scripts are removed, the main entry point can run the same pilot:

```bash
python scripts/run_experiments.py --app circlys --runs 1
```

For incomplete diagnostic analysis, use `python scripts/consolidate_results.py --evaluate --allow-partial`. The analysis reports incomplete coverage and does not label it as the completed final experiment.

## Saved outputs

| Path | Contents |
|---|---|
| `results/raw/<app>/<model>/<strategy>/run_01.json` through `run_05.json` | Individual run records |
| `results/logs/` | Execution logs, events and completion reports |
| `results/archive/` | Previous records when a result is replaced |
| `results/preflight/` | Full-RAG context checks |
| `results/consolidated/` | Combined predictions and run summaries in CSV/JSONL |
| `results/analysis/` | Evaluation tables and the false-positive heatmap in PNG/PDF |

Raw records retain prompts, responses, predictions, timestamps, runtime, settings, seeds, revisions, input hashes and environment details. RAG records retain retrieved excerpts and retrieval settings. Section records retain individual section predictions.

Incomplete parsing is recorded as `parse_failed`; missing predictions are not converted to zeros. Full-policy inputs are not silently truncated. The embedding model truncates embedding inputs at its configured token limit, and affected corpus chunk counts are recorded.

Keep and back up the complete `results/` folder, the expert-label workbook, inputs, configurations and exact code version. Generated outputs are excluded from Git. Later figures and alternative section aggregations can be calculated from saved results without rerunning the models.

## Evaluation

Evaluation is invoked explicitly with `consolidate_results.py --evaluate`; the inference runner does not invoke it automatically.

`main_table.csv` summarizes each model/strategy over complete repetitions, with metric means and sample standard deviations. Each repetition evaluates all configured applications together. Separate tables report each rubric criterion, criterion-macro performance, confusion counts, strategy contrasts and prediction stability. Undefined metrics remain blank.

OR is the primary aggregation for section strategies. Evaluation also compares **AND, strict majority and at least two positive sections**, using saved section predictions. These comparisons do not require new model inference and do not apply to full-policy strategies.

Additional outputs include application-cluster bootstrap confidence intervals, false-positive cases, descriptive paired-strategy and FP/FN tests, constant baselines and one false-positive heatmap. The exact item-level tests assume independent decisions; results should be interpreted alongside the application-cluster intervals. Additional figures can be built later from the saved records and tables.

Optional reviewed keyword patterns and independent annotator workbooks can be configured for baseline and expert-agreement analysis. The main workbook percentage column is not read: evaluation uses the individual binary criterion labels instead.
