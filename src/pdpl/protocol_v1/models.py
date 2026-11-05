from __future__ import annotations
import gc
import os
from typing import Optional

from .config import ModelSpec
from .retry import PermanentError
from .constraints import BinaryRubricSchema


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
    """Native chat formatting with model-selected binary values in a fixed rubric schema."""
    def __init__(self, spec: ModelSpec, hf_token: Optional[str] = None, rubric_codes=None):
        self.spec, self.hf_token = spec, hf_token
        self.tok = None
        self.mdl = None
        self._limit = None
        self.rubric_codes = list(rubric_codes or [])
        self._schema = None

    def render(self, prompt: str) -> str:
        tok = self.load_tokenizer()
        if tok.chat_template:
            return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
        if self.spec.prompt_template:
            return self.spec.prompt_template.format(prompt=prompt)
        raise PermanentError(f"No chat template available for {self.spec.hf_id}")

    def load_tokenizer(self):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(
                self.spec.hf_id, cache_dir=self.spec.cache_dir, use_fast=self.spec.tokenizer_use_fast,
                trust_remote_code=self.spec.trust_remote_code, revision=self.spec.revision,
                token=self.hf_token)
        return self.tok

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM
        self.load_tokenizer()
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_id, device_map=self.spec.device_map, torch_dtype=getattr(torch, self.spec.torch_dtype),
            cache_dir=self.spec.cache_dir, trust_remote_code=self.spec.trust_remote_code,
            revision=self.spec.revision, token=self.hf_token)
        return self

    def unload(self):
        self.mdl = None
        self.tok = None
        self._schema = None
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
                cfg = AutoConfig.from_pretrained(self.spec.hf_id, cache_dir=self.spec.cache_dir,
                                                 trust_remote_code=self.spec.trust_remote_code,
                                                 revision=self.spec.revision, token=self.hf_token)
            lim = getattr(cfg, "max_position_embeddings", None)
            if not lim:
                raise PermanentError(f"Cannot determine context limit for {self.spec.hf_id}; "
                                     f"set context_limit_tokens in configs/models.yaml")
            self._limit = int(lim)
        return self._limit

    def count_tokens(self, prompt: str) -> int:
        tok = self.load_tokenizer()
        return len(tok.encode(self.render(prompt), add_special_tokens=False))

    def generate(self, prompt, gen, seed):
        import torch
        text = self.render(prompt)
        inputs = self.tok(text, add_special_tokens=False, return_tensors="pt").to(self.mdl.device)
        if self._schema is None:
            self._schema = BinaryRubricSchema(self.tok, self.rubric_codes)
        if int(gen["max_new_tokens"]) < self._schema.max_tokens:
            raise PermanentError(f"max_new_tokens must be at least {self._schema.max_tokens} for the rubric schema")
        prompt_length = inputs["input_ids"].shape[1]
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        try:
            with torch.no_grad():
                out = self.mdl.generate(
                    **inputs, max_new_tokens=gen["max_new_tokens"], temperature=gen["temperature"],
                    top_p=gen["top_p"], repetition_penalty=gen["repetition_penalty"],
                    do_sample=gen["do_sample"], pad_token_id=self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id, num_beams=1, num_return_sequences=1,
                    prefix_allowed_tokens_fn=self._schema.callback(prompt_length))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            raise
        generated = out[0][prompt_length:].tolist()
        if not self._schema.is_complete(generated):
            raise PermanentError("Generation ended before all rubric values were selected")
        return self.tok.decode(generated, skip_special_tokens=True).strip()

    def info(self) -> dict:
        resolved = None
        if self.mdl is not None:
            resolved = getattr(self.mdl.config, "_commit_hash", None)
        return {"id": self.spec.id, "label": self.spec.label, "type": "hf_local", "hf_id": self.spec.hf_id,
                "revision_requested": self.spec.revision, "revision_resolved": resolved,
                "torch_dtype": self.spec.torch_dtype, "device_map": self.spec.device_map,
                "tokenizer_use_fast": self.spec.tokenizer_use_fast,
                "prompt_template": self.spec.prompt_template,
                "prompt_format": "tokenizer_chat_template" if self.tok is not None and self.tok.chat_template else "explicit_template",
                "chat_template": self.tok.chat_template if self.tok is not None else None,
                "output_protocol": BinaryRubricSchema.protocol,
                "output_codes": self.rubric_codes,
                "context_limit_tokens": self.context_limit() if (self.mdl is not None or self.spec.context_limit_tokens) else None}


def create_model(spec: ModelSpec, rubric_codes=None, hf_token_env: str = "HF_TOKEN") -> BaseModel:
    if spec.type != "hf_local":
        raise ValueError(f"Unsupported model type: {spec.type}")
    return HFLocalModel(spec, hf_token=os.environ.get(hf_token_env), rubric_codes=rubric_codes)
