# Behaviour changes relative to the notebook

Everything not listed here (prompts, rubric text and labels, 0/1 meaning, model IDs, sampling parameters, E5 embeddings, FAISS,
k = 3, parsing rules, OR aggregation, Arabic handling) is unchanged. Prompts, rubric, and parsers are verified against the notebook's own code
by `tests/test_notebook_fidelity.py`.

1. **Secret removed.** Cell 6 contained a real Hugging Face token. It is replaced in the notebook copy by a redaction string and read from the
   `HF_TOKEN` environment variable in the pipeline. *Revoke the exposed token.*
2. **Five runs per configuration** (notebook: one run). Sampling is stochastic (`do_sample=True`, temperature 0.1), so runs differ.
3. **`full_rag` = Experiment 1 "Full-Policy+Law".** The notebook's Experiment-1 cells (93, 96, …) were never executed (no stored outputs) and contain
   two defects: a stray literal `\n` after `pdpl_context = get_full_pdpl_context()` (a syntax error) and double-escaped regexes `r"\\b([01])\\b"`
   (they would not match digits). The repository uses the *intended* logic: the Exp. 1 prompt text and context retrieval
   (query = rubric text, k = 3, chunks joined by `"\n\n"`), with the baseline's parsers. Note the Exp. 1 prompt is not the baseline prompt plus context: as written in the
   notebook it omits the baseline's Arabic-handling sentences and the "output all 17 items" instruction. It is kept exactly as written.
4. **`section_norag` is new.** Its prompt is the `section_rag` prompt with the block `PDPL Excerpts:\n{pdpl_text}\n\n` removed and nothing else changed
   (tested). There is no notebook result to compare it with.
5. **Seeds.** The notebook was unseeded. Each generation is seeded with `seed_base + run*1000 + section_index` (`seed_base` in config; set `null` to disable).
   This makes runs reproducible per (run, section) where the hardware/library stack is deterministic; it does not change the sampling distribution.
6. **Failure handling.** Notebook: a section error silently became a vector of 17 zeros. Pipeline: transient errors are retried with exponential backoff; a section that still
   fails marks the **run** `failed` (saved, with the error), no zero vector is invented, and the run is retried on the next resume. The notebook's zero-padding of
   *under-long model output* (parsing) is kept and recorded (`padded_zeros`).
7. **Unparseable full-policy output.** Notebook: printed a warning and skipped. Pipeline: saved with status `parse_failed` (raw response kept), counted separately, retried on resume.
8. **Context-limit check (new).** The notebook had none. The pipeline checks `prompt tokens + max_new_tokens` against the model's `max_position_embeddings`
   (or `context_limit_tokens` in config) before every generation. If exceeded, nothing is truncated or sent; the run is recorded as `failed` with `ContextLimitExceeded` and not retried.
9. **Tokenizer `use_fast`.** The notebook's weight pre-download loop used `use_fast=True` for three models, but the loader that was actually used for inference used `use_fast=False`. Config uses `False` (the inference path).
10. **Loop order.** Notebook: app → model (reloading every model for every app). Pipeline: model → application → strategy → run, so each model is loaded once and released before the next
    (still strictly sequential; one (application, model) unit completes before the next starts). Results are unaffected.
11. **Retrieval output.** FAISS `similarity_search_with_score` is used instead of `retriever.invoke` to record L2 distances; ranking/top-k are identical. The embedder class now subclasses
    LangChain's `Embeddings` base (no numerical change; avoids a deprecation path).
12. **Model list.** The notebook header names Gemma-7B, but only four models have a loader/runner; those four are configured.
13. **Not ported:** metrics/plots/cross-app analysis/Experiments 2 and 3 (out of scope: "no new final statistical analyses"). `evaluation.py` keeps `evaluate` and `load_ground_truth` for later use.
14. **Preserved quirk:** the E5 embedder truncates inputs at 512 tokens (notebook behaviour). The number of truncated PDPL chunks is recorded in every RAG record (`retrieval.index.n_chunks_truncated_by_embedder`).
    This is embedding-side truncation only; LLM prompts are never truncated.
