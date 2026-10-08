from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .aggregation import aggregate
from .config import Application, Config, StrategySpec
from .models import BaseModel, ContextLimitExceeded
from .parsing import parse_full_response, parse_section_vector
from .policy import PolicyData
from .prompts import PromptTemplate, render_full_prompt, render_section_prompt
from .retrieval import Retriever
from .retry import retry_call
from .rubric import Rubric
from .utils import get_logger, utc_now_iso

log = get_logger("strategies")


@dataclass
class RunContext:
    cfg: Config
    app: Application
    policy: PolicyData
    model: BaseModel
    rubric: Rubric
    prompts: dict
    retriever: Optional[Retriever]
    sleep: Callable[[float], None] = time.sleep


def _err(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)[:2000],
            "attempts": getattr(exc, "attempts", 1),
            "permanent": True} 


def _generate(ctx: RunContext, prompt: str, seed: int) -> dict:
    """Context-limit check (never truncates) -> generation with retry/backoff."""
    gen = ctx.cfg.generation
    prompt_tokens = ctx.model.check_context(prompt, int(gen["max_new_tokens"]))
    t0 = time.time()

    def on_retry(attempt, exc, delay):
        log.warning("generation attempt %d failed (%s: %s); retrying in %.1fs", attempt, type(exc).__name__, exc, delay)

    raw, attempts = retry_call(lambda: ctx.model.generate(prompt, gen, seed), ctx.cfg.retry,
                               sleep=ctx.sleep, on_retry=on_retry)
    return {"raw_response": raw, "attempts": attempts, "generation_seconds": round(time.time() - t0, 3),
            "prompt_tokens": prompt_tokens, "context_limit_tokens": ctx.model.context_limit(),
            "max_new_tokens": int(gen["max_new_tokens"]), "seed": seed}


def _retrieval_record(query: str, k: int, sep: str, hits: List[dict], retriever: Retriever) -> dict:
    return {"query_chars": len(query), "top_k": k, "context_separator": sep,
            "retrieved": hits, "context_chars": len(sep.join(h["text"] for h in hits)),
            "index": retriever.info()}


def run_section_strategy(spec: StrategySpec, ctx: RunContext, run: int) -> dict:
    r, rub, cfg = ctx.retriever, ctx.rubric, ctx.cfg
    n_items = len(rub)
    template: PromptTemplate = ctx.prompts[spec.name]
    sections_out = []
    failed = None
    for i, section in enumerate(ctx.policy.sections, start=1):
        seed = int(cfg.generation["seed_base"]) + run * 1000 + i if cfg.generation.get("seed_base") is not None else None
        entry = {"section_index": i, "title": section["title"], "content_chars": len(section["content"]),
                 "status": "success", "seed": seed}
        t0 = time.time()
        try:
            pdpl_text = None
            if spec.rag:
                k = int(cfg.retrieval["top_k_section"])
                sep = cfg.retrieval["section_context_separator"]
                hits = r.retrieve(section["content"], k)
                pdpl_text = sep.join(h["text"] for h in hits)
                entry["retrieval"] = _retrieval_record(section["content"], k, sep, hits, r)
            prompt = render_section_prompt(template, rub.text, section["title"], section["content"], pdpl_text)
            entry["prompt"] = prompt
            g = _generate(ctx, prompt, seed)
            entry.update(g)
            vec, meta = parse_section_vector(g["raw_response"], rub.items)
            entry["parsed_vector"] = vec
            entry["parse"] = meta
            if vec is None:
                entry.update(status="parse_failed", error={"type": "ParseError", "message": "section output has an incorrect prediction count", "details": meta})
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = _err(e)
            entry["parsed_vector"] = None
            failed = (i, e)
        entry["section_seconds"] = round(time.time() - t0, 3)
        sections_out.append(entry)
        if entry["status"] == "parse_failed":
            return {"status": "parse_failed", "prediction": None, "section_predictions": sections_out,
                    "aggregation": {"rule": spec.aggregation}, "error": entry["error"]}
        if failed:
            break
    out = {"section_predictions": sections_out,
           "aggregation": {"rule": spec.aggregation, "notebook_equivalent": "np.array(vectors).max(axis=0)"}}
    if failed:
        i, e = failed
        out.update(status="failed", prediction=None,
                   error={**_err(e), "message": f"section {i}/{ctx.policy.n_sections}: {e}"[:2000]})
        return out
    final = aggregate([s["parsed_vector"] for s in sections_out], spec.aggregation)
    out.update(status="success", prediction={item: int(v) for item, v in zip(rub.items, final)}, error=None)
    return out


def run_full_strategy(spec: StrategySpec, ctx: RunContext, run: int) -> dict:
    cfg, rub = ctx.cfg, ctx.rubric
    template: PromptTemplate = ctx.prompts[spec.name]
    seed = int(cfg.generation["seed_base"]) + run * 1000 if cfg.generation.get("seed_base") is not None else None
    out: dict = {"seed": seed}
    try:
        pdpl_context = None
        if spec.rag:
            k = int(cfg.retrieval["top_k_full"])
            sep = cfg.retrieval["full_context_separator"]
            hits = ctx.retriever.retrieve(rub.text, k) 
            pdpl_context = sep.join(h["text"] for h in hits) 
            out["retrieval"] = _retrieval_record(rub.text, k, sep, hits, ctx.retriever)
        prompt = render_full_prompt(template, rub.text, ctx.policy.full_text, pdpl_context)
        out["prompt"] = prompt
        out["token_lengths"] = {
            "policy_tokens": ctx.model.count_tokens(ctx.policy.full_text),
            "retrieved_context_tokens": ctx.model.count_tokens(pdpl_context) if pdpl_context else 0,
            "prompt_tokens": ctx.model.count_tokens(prompt),
            "max_new_tokens": int(cfg.generation["max_new_tokens"]),
            "context_limit_tokens": ctx.model.context_limit(),
            "token_counter": "model tokenizer (inputs rendered with the model prompt template)"}
        out["token_lengths"]["total_tokens"] = out["token_lengths"]["prompt_tokens"] + out["token_lengths"]["max_new_tokens"]
        g = _generate(ctx, prompt, seed)
        out.update(g)
    except Exception as e:
        out.update(status="failed", prediction=None, error=_err(e))
        return out
    preds, meta = parse_full_response(g["raw_response"], rub.items)
    out["parse"] = meta
    if preds is None:
        out.update(status="parse_failed", prediction=None,
                   error={"type": "ParseError", "message": "output did not contain a complete set of binary predictions",
                          "attempts": g["attempts"], "permanent": False})
    else:
        out.update(status="success", prediction=preds, error=None)
    return out


def execute_strategy(spec: StrategySpec, ctx: RunContext, run: int) -> dict:
    if spec.rag and ctx.retriever is None:
        raise RuntimeError(f"Strategy {spec.name} needs a retriever")
    fn = run_section_strategy if spec.unit == "section" else run_full_strategy
    started = utc_now_iso()
    t0 = time.time()
    res = fn(spec, ctx, run)
    res["started_at"], res["finished_at"] = started, utc_now_iso()
    res["runtime_seconds"] = round(time.time() - t0, 3)
    return res
