"""Prompt templates (verbatim notebook prompts stored in prompts/*.txt) and rendering."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import sha256_text


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    path: str
    text: str
    sha256: str

    def render(self, **kwargs) -> str:
        return self.text.format(**kwargs)


def load_prompt(name: str, path: str | Path) -> PromptTemplate:
    text = Path(path).read_text(encoding="utf-8")
    return PromptTemplate(name=name, path=str(path), text=text, sha256=sha256_text(text))


def render_section_prompt(t: PromptTemplate, rubric_text: str, title: str, content: str,
                          pdpl_text: str | None = None) -> str:
    """Notebook: build_section_query(section, pdpl_text). pdpl_text is ignored by the no-RAG template."""
    return t.render(rubric_text=rubric_text, section_title=title, section_content=content,
                    pdpl_text=pdpl_text or "")


def render_full_prompt(t: PromptTemplate, rubric_text: str, full_policy_text: str,
                       pdpl_context: str | None = None) -> str:
    """Notebook: build_full_policy_prompt / build_full_policy_grounded_prompt (Experiment 1)."""
    return t.render(rubric_text=rubric_text, full_policy_text=full_policy_text,
                    pdpl_context=pdpl_context or "")
