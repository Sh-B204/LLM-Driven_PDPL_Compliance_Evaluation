from __future__ import annotations
from typing import Callable, Optional
from .config import Application, Config, ModelSpec
from .models import BaseModel, create_model
from .policy import load_policy
from .prompts import load_prompt, render_full_prompt
from .rubric import load_rubric
from .runner import build_cfg_retriever


def longest_policy(cfg: Config) -> Optional[Application]:
    """Application whose full policy text (as sent to the model) is longest, among policy files that exist."""
    best, best_len = None, -1
    for a in cfg.applications:
        if a.policy_path.exists():
            n = len(load_policy(a.policy_path).full_text)
            if n > best_len:
                best, best_len = a, n
    return best


def preflight_full_rag(cfg: Config, app: Application, model_spec: ModelSpec,
                       model_factory: Optional[Callable[[ModelSpec, object], BaseModel]] = None) -> dict:
    """Measure policy / retrieved-context / prompt / total token lengths for strategy full_rag and
    compare with the model's context limit. Loads only the tokenizer + config (not the weights)."""
    rubric = load_rubric(cfg.rubric_path)
    spec = cfg.strategies["full_rag"]
    policy = load_policy(app.policy_path)
    retr = build_cfg_retriever(cfg)
    try:
        hits = retr.retrieve(rubric.text, int(cfg.retrieval["top_k_full"]))
    finally:
        retr.embedder.release()
    context = cfg.retrieval["full_context_separator"].join(h["text"] for h in hits)
    prompt = render_full_prompt(load_prompt("full_rag", spec.prompt_path), rubric.text, policy.full_text, context)
    model = (model_factory or (lambda s, r: create_model(s, rubric_codes=r.codes,
                                                          hf_token_env=cfg.output.get("hf_token_env", "HF_TOKEN"))))(model_spec, rubric)
    model.load_tokenizer()
    limit = model.context_limit()
    max_new = int(cfg.generation["max_new_tokens"])
    lengths = {"policy_tokens": model.count_tokens(policy.full_text),
               "retrieved_context_tokens": model.count_tokens(context),
               "prompt_tokens": model.count_tokens(prompt), "max_new_tokens": max_new}
    lengths["total_tokens"] = lengths["prompt_tokens"] + max_new
    ok = lengths["total_tokens"] <= limit
    return {"application": app.id, "model": model_spec.id, "strategy": "full_rag",
            "policy_chars": len(policy.full_text), "retrieved_context_chars": len(context),
            "prompt_chars": len(prompt), "token_lengths": lengths, "context_limit_tokens": limit,
            "fits_context": ok,
            "reason": None if ok else (f"prompt ({lengths['prompt_tokens']}) + max_new_tokens ({max_new}) = "
                                       f"{lengths['total_tokens']} exceeds the model context limit ({limit}); "
                                       f"nothing was truncated and no request was sent")}
