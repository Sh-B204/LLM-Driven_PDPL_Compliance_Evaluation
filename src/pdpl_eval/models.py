from __future__ import annotations
import gc
import os
from typing import Optional

from .config import ModelSpec
from .retry import PermanentError


class ContextLimitExceeded(PermanentError):
    def __init__(self, prompt_tokens: int, max_new_tokens: int, limit: int):
        self.prompt_tokens, self.max_new_tokens, self.limit = prompt_tokens, max_new_tokens, limit
        super().__init__(f"Request needs {prompt_tokens} prompt + {max_new_tokens} new tokens = "
                         f"{prompt_tokens + max_new_tokens} but the model context limit is {limit}. "
                         f"Nothing was truncated; the request was not sent.")


class BaseModel:
    spec: ModelSpec

    def render(self, prompt: str) -> str:
        t = self.spec.prompt_template
        return t.format(prompt=prompt) if t else prompt

    def load_tokenizer(self):
        raise NotImplementedError
    def load(self):
        raise NotImplementedError
    def unload(self):
        raise NotImplementedError
    def count_tokens(self, prompt: str) -> int:
        raise NotImplementedError
    def context_limit(self) -> int:
        raise NotImplementedError
    def generate(self, prompt: str, gen: dict, seed: Optional[int]) -> str:
        raise NotImplementedError
    def info(self) -> dict:
        raise NotImplementedError

    def check_context(self, prompt: str, max_new_tokens: int) -> int:
        n = self.count_tokens(prompt)
        limit = self.context_limit()
        if n + max_new_tokens > limit:
            raise ContextLimitExceeded(n, max_new_tokens, limit)
        return n


class HFLocalModel(BaseModel):
    """Notebook load_hf_model + run_plain, same arguments."""
    def __init__(self, spec: ModelSpec, hf_token: Optional[str] = None):
        self.spec, self.hf_token = spec, hf_token
        self.tok = None
        self.mdl = None
        self._limit = None
        self._logged_in = False

    def _login(self):
        if self.hf_token and not self._logged_in:
            from huggingface_hub import login
            login(self.hf_token)
            self._logged_in = True

    def load_tokenizer(self):
        if self.tok is None:
            from transformers import AutoTokenizer
            self._login()
            self.tok = AutoTokenizer.from_pretrained(
                self.spec.hf_id, cache_dir=self.spec.cache_dir, use_fast=self.spec.tokenizer_use_fast,
                trust_remote_code=self.spec.trust_remote_code, revision=self.spec.revision)
        return self.tok

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM
        self.load_tokenizer()
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_id, device_map=self.spec.device_map, torch_dtype=getattr(torch, self.spec.torch_dtype),
            cache_dir=self.spec.cache_dir, trust_remote_code=self.spec.trust_remote_code,
            revision=self.spec.revision)
        return self

    def unload(self):
        self.mdl = None
        self.tok = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def context_limit(self) -> int:
        if self.spec.context_limit_tokens:
            return int(self.spec.context_limit_tokens)
        if self._limit is None:
            if self.mdl is not None:
                cfg = self.mdl.config
            else:
                from transformers import AutoConfig
                self._login()
                cfg = AutoConfig.from_pretrained(self.spec.hf_id, cache_dir=self.spec.cache_dir,
                                                 trust_remote_code=self.spec.trust_remote_code,
                                                 revision=self.spec.revision)
            lim = getattr(cfg, "max_position_embeddings", None)
            if not lim:
                raise PermanentError(f"Cannot determine context limit for {self.spec.hf_id}; "
                                     f"set context_limit_tokens in configs/models.yaml")
            self._limit = int(lim)
        return self._limit

    def count_tokens(self, prompt: str) -> int:
        tok = self.load_tokenizer()
        return int(tok(self.render(prompt), return_tensors="pt")["input_ids"].shape[1])

    def generate(self, prompt, gen, seed):
        import torch
        text = self.render(prompt)
        inputs = self.tok(text, return_tensors="pt").to(self.mdl.device)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        try:
            with torch.no_grad():
                out = self.mdl.generate(
                    **inputs, max_new_tokens=gen["max_new_tokens"], temperature=gen["temperature"],
                    top_p=gen["top_p"], repetition_penalty=gen["repetition_penalty"],
                    do_sample=gen["do_sample"], pad_token_id=self.tok.eos_token_id)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            raise
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def info(self) -> dict:
        resolved = None
        if self.mdl is not None:
            resolved = getattr(self.mdl.config, "_commit_hash", None)
        return {"id": self.spec.id, "label": self.spec.label, "type": "hf_local", "hf_id": self.spec.hf_id,
                "revision_requested": self.spec.revision, "revision_resolved": resolved,
                "torch_dtype": self.spec.torch_dtype, "device_map": self.spec.device_map,
                "tokenizer_use_fast": self.spec.tokenizer_use_fast,
                "prompt_template": self.spec.prompt_template,
                "context_limit_tokens": self.context_limit() if (self.mdl is not None or self.spec.context_limit_tokens) else None}


def create_model(spec: ModelSpec, rubric_codes=None, hf_token_env: str = "HF_TOKEN") -> BaseModel:
    if spec.type != "hf_local":
        raise ValueError(f"Unsupported model type: {spec.type}")
    return HFLocalModel(spec, hf_token=os.environ.get(hf_token_env))
