from __future__ import annotations
import time
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import __version__
from .config import Application, Config, ModelSpec, Selection, STRATEGY_NOTEBOOK_LABELS, StrategySpec, select
from .environment import environment_info
from .models import BaseModel, create_model
from .policy import PolicyData, load_policy
from .prompts import load_prompt
from .retrieval import Retriever, build_retriever
from .rubric import Rubric, load_rubric
from .storage import SCHEMA_VERSION, append_event, raw_path, save_result, verify_saved
from .strategies import RunContext, execute_strategy
from .utils import get_logger, utc_now_iso
from .validation import is_valid_completed
from .storage import load_json
from .utils import fingerprint, sha256_file

log = get_logger("runner")

PARSER_RULES = {
    "section": "first isolated binary value per line; exact rubric length; no padding/truncation",
    "full": "notebook code-anchored extraction; all criteria required; exact-length numerical fallback",
}




def experiment_fingerprint(cfg, app, model_spec, spec, rubric, template):
    """Portable content identity; excludes output paths and the selected run count."""
    model = asdict(model_spec)
    model.pop("cache_dir", None)
    settings = {"policy_sha256": sha256_file(app.policy_path), "rubric_sha256": rubric.sha256,
                "prompt_sha256": template.sha256, "model": model, "generation": dict(cfg.generation),
                "strategy": {"name": spec.name, "unit": spec.unit, "rag": spec.rag, "aggregation": spec.aggregation},
                "parser": PARSER_RULES[spec.unit], "pipeline_version": __version__}
    sources = ["policy.py", "parsing.py", "aggregation.py", "strategies.py", "models.py", "prompts.py"]
    if spec.rag:
        sources.append("retrieval.py")
        retrieval = dict(cfg.retrieval)
        corpus = Path(retrieval.pop("corpus_path"))
        corpus = corpus if corpus.is_absolute() else cfg.root / corpus
        settings["retrieval"] = retrieval
        settings["corpus_sha256"] = sha256_file(corpus)
    settings["code_sha256"] = {name: sha256_file(Path(__file__).parent / name) for name in sources}
    return fingerprint(settings)


def can_reuse_result(path, expected, revision, embedding_revision=None):
    if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision) or not is_valid_completed(path):
        return False
    rec = load_json(path)
    if rec.get("rag") and not re.fullmatch(r"[a-fA-F0-9]{40}", embedding_revision or ""):
        return False
    return rec.get("experiment_fingerprint") == expected and rec["model"].get("revision_resolved") == revision


def build_record(cfg: Config, app: Application, policy: PolicyData, model_info: dict, spec: StrategySpec,
                 run: int, rubric: Rubric, template, result: dict, env: dict) -> dict:
    params = {
        "generation": dict(cfg.generation),
        "retry": dict(cfg.retry),
        "prompt": {"path": template.path, "sha256": template.sha256},
        "model_prompt_template": model_info.get("prompt_template"),
        "parser": PARSER_RULES[spec.unit],
        "seed": result.get("seed"),
    }
    if spec.rag:
        params["retrieval"] = {k: cfg.retrieval[k] for k in
                               ("embedding_model", "embedding_max_length", "chunking", "corpus_path",
                                "top_k_section" if spec.unit == "section" else "top_k_full",
                                "section_context_separator" if spec.unit == "section" else "full_context_separator")}
        params["retrieval"]["query"] = "section content" if spec.unit == "section" else "rubric text (all items)"
    rec = {
        "schema_version": SCHEMA_VERSION, "pipeline_version": __version__,
        "application": {"id": app.id, "name": app.name},
        "model": model_info,
        "strategy": spec.name, "strategy_notebook_label": STRATEGY_NOTEBOOK_LABELS[spec.name],
        "rag": spec.rag, "unit": spec.unit, "run": run,
        "policy": {"path": str(app.policy_path), "sha256": policy.sha256, "n_sections": policy.n_sections,
                   "full_text_chars": len(policy.full_text)},
        "rubric": {"path": str(cfg.rubric_path), "sha256": rubric.sha256, "items": rubric.items, "codes": rubric.codes},
        "parameters": params, "environment": env,
    }
    model_spec = next(m for m in cfg.models if m.id == model_info["id"])
    rec["experiment_fingerprint"] = experiment_fingerprint(cfg, app, model_spec, spec, rubric, template)
    rec["configuration_snapshot"] = {"generation": dict(cfg.generation), "retry": dict(cfg.retry),
                                     "retrieval": dict(cfg.retrieval) if spec.rag else None,
                                     "model": {k: v for k, v in asdict(model_spec).items() if k != "cache_dir"}}
    rec.update(result)
    return rec


def build_cfg_retriever(cfg: Config) -> Retriever:
    corpus = Path(cfg.retrieval["corpus_path"])
    corpus = corpus if corpus.is_absolute() else cfg.root / corpus
    return build_retriever({**cfg.retrieval, "corpus_path": str(corpus)})


def _new_summary() -> dict:
    return {"completed": [], "failed": [], "parse_failed": [], "skipped": [], "blocked": [], "save_errors": [],
            "interrupted": False}


def plan(cfg: Config, sel: Selection) -> List[tuple]:
    return [(m, a, s, r) for m in sel.models for a in sel.applications for s in sel.strategies for r in sel.run_numbers]


def run_experiments(cfg: Config, sel: Selection, force: bool = False, dry_run: bool = False,
                    sleep: Callable[[float], None] = time.sleep,
                    model_factory: Optional[Callable[[ModelSpec, Rubric], BaseModel]] = None) -> dict:
    results_dir = cfg.results_dir
    summary = _new_summary()
    rubric = load_rubric(cfg.rubric_path)
    prompts = {s.name: load_prompt(s.name, s.prompt_path) for s in sel.strategies}
    model_factory = model_factory or (lambda spec, rub: create_model(spec, rubric_codes=rub.codes,
                                                                     hf_token_env=cfg.output.get("hf_token_env", "HF_TOKEN")))
    env = None
    retriever_holder: Dict[str, Optional[Retriever]] = {"r": None, "err": None}
    policy_cache: Dict[str, object] = {}

    def get_policy(app: Application):
        if app.id not in policy_cache:
            try:
                policy_cache[app.id] = load_policy(app.policy_path)
            except Exception as e:
                policy_cache[app.id] = e
        return policy_cache[app.id]

    def get_retriever():
        if retriever_holder["r"] is None and retriever_holder["err"] is None:
            try:
                log.info("Building FAISS index (%s)", cfg.retrieval["embedding_model"])
                retriever_holder["r"] = build_cfg_retriever(cfg)
            except Exception as e:
                retriever_holder["err"] = e
                log.error("Retriever unavailable: %s", e)
        return retriever_holder["r"]

    def block(m, a, s, r, reason):
        summary["blocked"].append({"model": m.id, "app": a.id, "strategy": s.name, "run": r, "reason": reason})

    try:
        for model_spec in sel.models:
            if not dry_run and not re.fullmatch(r"[a-fA-F0-9]{40}", model_spec.revision or ""):
                for a in sel.applications:
                    for s in sel.strategies:
                        for r in sel.run_numbers:
                            block(model_spec, a, s, r, "Pin an immutable model revision before real execution")
                continue
            pending = []
            for a in sel.applications:
                for s in sel.strategies:
                    for r in sel.run_numbers:
                        p = raw_path(results_dir, a.id, model_spec.id, s.name, r)
                        if s.rag and not dry_run and not re.fullmatch(r"[a-fA-F0-9]{40}", cfg.retrieval.get("embedding_revision") or ""):
                            block(model_spec, a, s, r, "Pin an immutable embedding revision before RAG execution")
                            continue
                        try:
                            expected = experiment_fingerprint(cfg, a, model_spec, s, rubric, prompts[s.name])
                        except Exception as e:
                            block(model_spec, a, s, r, f"input validation failed: {e}")
                            continue
                        if not force and can_reuse_result(p, expected, model_spec.revision, cfg.retrieval.get("embedding_revision")):
                            summary["skipped"].append({"model": model_spec.id, "app": a.id, "strategy": s.name, "run": r})
                        else:
                            pending.append((a, s, r, p))
            log.info("Model %s: %d pending, skipped so far %d", model_spec.id, len(pending), len(summary["skipped"]))
            if dry_run or not pending:
                if dry_run:
                    for a, s, r, p in pending:
                        log.info("[dry-run] would run %s / %s / %s / run_%02d", a.id, model_spec.id, s.name, r)
                continue
            env = env or environment_info()
            model = model_factory(model_spec, rubric)
            try:
                log.info("Loading model %s (%s)", model_spec.label, model_spec.hf_id)
                model.load()
            except Exception as e:
                log.error("Model %s failed to load: %s", model_spec.id, e)
                for a, s, r, p in pending:
                    block(model_spec, a, s, r, f"model load failed: {type(e).__name__}: {e}")
                try:
                    model.unload()
                except Exception:
                    pass
                continue
            try:
                model_info = model.info()
                for a in sel.applications:
                    unit = [(s, r, p) for (aa, s, r, p) in pending if aa.id == a.id]
                    if not unit:
                        continue
                    pol = get_policy(a)
                    if isinstance(pol, Exception):
                        for s, r, p in unit:
                            block(model_spec, a, s, r, f"policy unavailable: {pol}")
                        log.error("Application %s skipped: %s", a.id, pol)
                        continue
                    log.info("=== unit: application=%s model=%s (%d runs to do)", a.id, model_spec.id, len(unit))
                    for s, r, p in unit:
                        retr = None
                        if s.rag:
                            retr = get_retriever()
                            if retr is None:
                                block(model_spec, a, s, r, f"retriever unavailable: {retriever_holder['err']}")
                                continue
                        ctx = RunContext(cfg, a, pol, model, rubric, prompts, retr, sleep=sleep)
                        res = execute_strategy(s, ctx, r)
                        rec = build_record(cfg, a, pol, model_info, s, r, rubric, prompts[s.name], res, env)
                        existed = p.exists()
                        try:
                            archived = save_result(results_dir, p, rec, allow_replace_existing=existed)
                        except Exception as e:
                            log.error("SAVE FAILED %s: %s", p, e)
                            summary["save_errors"].append({"path": str(p), "problems": [str(e)]})
                            continue
                        problems = verify_saved(p, rec)
                        key = {"model": model_spec.id, "app": a.id, "strategy": s.name, "run": r}
                        if problems:
                            log.error("VERIFY FAILED %s: %s", p, problems)
                            summary["save_errors"].append({"path": str(p), "problems": problems})
                            append_event(results_dir, {**key, "ts": utc_now_iso(), "event": "verify_failed", "problems": problems})
                            continue
                        summary[{"success": "completed"}.get(res["status"], res["status"])].append(key)
                        append_event(results_dir, {**key, "ts": utc_now_iso(), "event": res["status"], "path": str(p),
                                                   "runtime_seconds": res["runtime_seconds"],
                                                   "archived_previous": str(archived) if archived else None,
                                                   "error": res.get("error")})
                        log.info("%s | %s | %s | run_%02d -> %s (%.1fs) saved+verified",
                                 a.id, model_spec.id, s.name, r, res["status"], res["runtime_seconds"])
            finally:
                model.unload()
                log.info("Model %s unloaded", model_spec.id)
    except KeyboardInterrupt:
        summary["interrupted"] = True
        log.warning("Interrupted. Completed runs are saved; re-run the same command to resume.")
    finally:
        if retriever_holder["r"] is not None:
            try:
                retriever_holder["r"].embedder.release()
            except Exception:
                pass
    return summary


def check_inputs(cfg):
    """Validate local inputs and import every module; no network or weights required."""
    import importlib
    import pkgutil
    import pdpl_eval
    from .evaluation import load_labels
    from .policy import parse_docx_to_sections
    from .prompts import render_full_prompt, render_section_prompt
    for module in pkgutil.iter_modules(pdpl_eval.__path__):
        importlib.import_module(f"pdpl_eval.{module.name}")
    rubric = load_rubric(cfg.rubric_path)
    labels, path = load_labels(cfg, rubric)
    policies = [load_policy(a.policy_path) for a in cfg.applications]
    if any(not x.full_text.strip() or not x.sections for x in policies):
        raise ValueError("Empty policy text/sections")
    corpus = cfg.root / cfg.retrieval["corpus_path"]
    if not parse_docx_to_sections(corpus):
        raise ValueError("Empty law corpus")
    for spec in cfg.strategies.values():
        t = load_prompt(spec.name, spec.prompt_path)
        if spec.unit == "section":
            render_section_prompt(t, rubric.text, "Title", "Content", "Law" if spec.rag else None)
        else:
            render_full_prompt(t, rubric.text, "Content", "Law" if spec.rag else None)
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb[cfg.ground_truth.get("sheet_name", "Expert Analysis")].values)
        headers = {re.match(r"^\s*([A-D]\d+)\s*\.", str(h or "")).group(1): h for h in rows[0]
                   if re.match(r"^\s*([A-D]\d+)\s*\.", str(h or ""))}
        normalize = lambda name: " ".join(str(name).split()).casefold()
        aliases = cfg.ground_truth.get("application_aliases", {})
        included = {normalize(aliases.get(a.id, a.name)) for a in cfg.applications}
        extra = [str(row[0]) for row in rows[1:] if row[0] and normalize(row[0]) not in included]
    finally:
        wb.close()
    report = {"included_ids": [a.id for a in cfg.applications], "excluded_workbook_rows": extra,
              "article_mapping_from_workbook": {code: headers[code] for code in rubric.codes},
              "applications": len(policies), "criteria": len(rubric), "expert_labels": len(labels),
              "expected_runs": len(plan(cfg, select(cfg))),
              "ground_truth_sha256": sha256_file(path), "law_sha256": sha256_file(corpus),
              "policies": [{"id": a.id, "sha256": x.sha256, "n_sections": x.n_sections} for a, x in zip(cfg.applications, policies)],
              "real_generation_tested": False}
    from .storage import atomic_write_json
    atomic_write_json(cfg.results_dir / "logs" / "input_check.json", report)
    return report


def pin_model_revisions(cfg):
    """Resolve immutable revisions through metadata only; retain YAML formatting/comments."""
    import os
    from huggingface_hub import HfApi
    from .storage import atomic_write_text
    api = HfApi(token=os.environ.get(cfg.output.get("hf_token_env", "HF_TOKEN")))
    model_path = cfg.root / cfg.raw["models_file"]
    text = model_path.read_text(encoding="utf-8")
    for model in cfg.models:
        revision = model.revision
        if not re.fullmatch(r"[a-fA-F0-9]{40}", revision or ""):
            revision = api.model_info(model.hf_id, revision=revision or "main").sha
        pattern = r'(^  - id: "?' + re.escape(model.id) + r'"?\s*\n[\s\S]*?^    revision:) [^\n#]*'
        text, count = re.subn(pattern, lambda m: m.group(1) + ' "' + revision + '" ', text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Cannot locate revision field for {model.id}; no files changed")
        print(model.id, revision)
    retrieval = cfg.retrieval
    revision = retrieval.get("embedding_revision")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision or ""):
        revision = api.model_info(retrieval["embedding_model"], revision=revision or "main").sha
    exp = cfg.source_file.read_text(encoding="utf-8")
    if re.search(r"^  embedding_revision:", exp, re.MULTILINE):
        exp = re.sub(r"^  embedding_revision:.*$", '  embedding_revision: "' + revision + '"', exp, flags=re.MULTILINE)
    else:
        exp = exp.replace("retrieval:\n", 'retrieval:\n  embedding_revision: "' + revision + '"\n', 1)
    atomic_write_text(model_path, text)
    atomic_write_text(cfg.source_file, exp)
    print("Model and embedding revisions pinned; no weights downloaded")
