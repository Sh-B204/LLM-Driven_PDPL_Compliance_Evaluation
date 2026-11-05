# LLM-Driven PDPL Disclosure Evaluation

Evaluate 21 Saudi FinTech policy documents against 15 binary disclosure criteria using four models and four strategies (section/full policy, with/without RAG), with five repetitions. Predictions concern disclosure in policy text, not implemented protections or legal compliance.

## Files

| Path | Purpose |
|---|---|
| `configs/` | Input paths, pinned model revisions and settings |
| `data/` | The 21 policy documents, regulatory corpus and final expert reference labels |
| `prompts/`, `rubric/` | Four prompt templates and 15-item rubric |
| `src/pdpl/` | Current inference package (protocol 2.0.0) |
| `src/pdpl/protocol_v1/` | Original protocol 1.1.0 code required to verify saved paper records |
| `results/raw/` | Saved paper records; preserve unchanged |
| `provenance/` | Checksum manifests |
| `evaluation/` | Current combined evaluation tables and figures |
| `input_analysis/` | Input statistics, extraction audit and token measurements |

## Install and validate

Use Python 3.10 or newer. Install a GPU-compatible PyTorch build for new inference, then:

```bash
python -m pip install -r requirements.txt
python run_experiments.py --check-inputs
```

Set `HF_TOKEN` privately for gated models. Scripts read the environment variable; they do not automatically load `.env` files. Never commit a real token.

## Reproduce the paper from saved records

No language-model inference or model downloads are needed:

```bash
python evaluate_results.py --repo . --results results --out evaluation --boot 1000
python plot_results.py --evaluation evaluation --repo . --check-only
python plot_results.py --evaluation evaluation --repo .
python analyze_inputs.py --repo . --results results --out input_analysis
```

Evaluation writes CSV/XLSX tables. Plotting writes PNGs and native LaTeX tables including `T_available`, `T_asymmetry`, `T_criterion_errors` and `T_rag_errors`. Per-application red/green inspection grids are generated in `evaluation/figures/application_binary_grids/`; `--skip-binary` skips them without deleting existing grids. Input analysis writes PNG figures only. Historical PDFs can be removed.

OR is the main section rule. `--aggregation at_least_two`, `majority` or `and` produces exploratory figures in `figures_<rule>/`. RAG comparisons use matching policies within each model and input type.

Basic input analysis uses saved token measurements. To regenerate current measurements for all 21 documents, including Drahim's unrequested ALLaM full-policy input, append `--tokenizers --embeddings`. This downloads tokenizers and loads the small embedding model, not the four language models. Word counts are descriptive; context feasibility uses model-token counts.

## Final study coverage

The 21 x 4 x 4 x 5 design has 1,680 potential records. Ten Drahim/ALLaM full-policy records were not requested. All 1,670 saved records verify as archived protocol 1.1.0: 1,540 successful and 130 context-limit failures. ALLaM covers 20 policies under both section strategies, 11 under full policy without RAG and nine with RAG. Qwen and Mistral cover 21 under section input and 20 under full-policy input. LLaMA covers all 21 under every strategy. Eight configurations cover all 21; nine policies are common to all configurations. Failed predictions are excluded, never scored as zero.

Awaed uses the corrected document and matching saved predictions. ALLaM section-RAG and Qwen full policy without RAG both reach 66.67% accuracy on the same 20 policies. Pooled repetition counts are repeated predictions on the same policies, not independent observations.

## Running new experiments

Check the RAG context, run the pilot, then start the full experiment:

```bash
python run_pilot.py --app circlys --preflight-only
python run_pilot.py --app circlys
python run_experiments.py --resume
```

New runs are saved to `results_v2/` using updated prompting and output constraints. To reproduce the paper’s reported results, use the saved records in `results/raw/` with the evaluation commands above.
