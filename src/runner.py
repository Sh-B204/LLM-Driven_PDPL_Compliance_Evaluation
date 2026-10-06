"""Sequential experiment runner: model -> application -> strategy -> run. Save-verify-continue; resumable."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import __version__
from .config import Application, Config, ModelSpec, Selection, STRATEGY_NOTEBOOK_LABELS, StrategySpec
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

log = get_logger("runner")

PARSER_RULES = {
    "section": "first isolated 0/1 per output line, in order; zero-padded/truncated to the rubric length "
               "(notebook cell 15); aggregation applied afterwards",
    "full": "code-anchored regex parse_llm_response (notebook cell 10); fallback = last N isolated 0/1 values; "
            "else status parse_failed",
}


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
            pending = []
            for a in sel.applications:
                for s in sel.strategies:
                    for r in sel.run_numbers:
                        p = raw_path(results_dir, a.id, model_spec.id, s.name, r)
                        if p.exists() and is_valid_completed(p) and not force:
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
