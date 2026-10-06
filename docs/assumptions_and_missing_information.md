# Assumptions and missing information

* **No real data was available** to the author of this repository (policies, `IRPDPL.docx`, ground-truth workbook, GPU, model weights). Everything was verified with the mock model on synthetic data and
  against the notebook's code; **no real-model run has been executed**. The first real run should be `run_smoke_test.py`, then `run_full_policy_rag_test.py`.
* Policy files are expected at `data/policies/<Application Name>.docx` (notebook: `user pp/preprocessed/<Name>.docx`); change `policy_path` in `configs/applications.yaml` if different.
* The notebook cross-app cell lists 19 applications; the app cells (15…69) confirm the same 19 names and policy file names. Application IDs are lowercase slugs (e.g. `tamra_capital`).
* "Full policy with RAG" is interpreted as Experiment 1 (top-3 PDPL chunks retrieved with the whole rubric as query). "Section-by-section without RAG" has no notebook equivalent (see behaviour changes §4).
* Context limits come from each model's `max_position_embeddings`; ALLaM's config value is assumed to be present (otherwise set `context_limit_tokens`). This is the architectural limit, not a guarantee that a 7–8B model reads very long inputs well or that GPU memory suffices.
* GPU memory is not pre-estimated. A CUDA OOM is treated as transient (cache cleared, retried); repeated OOM is recorded as a failed run.
* Hugging Face model revisions were not pinned in the notebook. `revision: null` follows the latest snapshot; the resolved commit hash is stored per record where available. Pin revisions in `configs/models.yaml` for strict reproducibility.
* Gated models (e.g. Llama-3.1) require accepting the licence on Hugging Face and a valid `HF_TOKEN`.
* One process at a time should write to a results directory (no cross-process lock is implemented); overwritten files are archived, never deleted.
* Section parsing relies on Word "Heading" styles. Policies without heading styles become a single "Introduction" section (same as the notebook).
* Seed determinism depends on hardware, CUDA, and library versions; package versions and GPU name are stored in every record.
