from __future__ import annotations
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import platform
import sys
from importlib import metadata

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def fingerprint(obj) -> str:
    return sha256_text(canonical_json(obj))


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def safe_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def setup_logging(log_dir: str | Path, name: str, verbose: bool = True) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{utc_stamp()}.log"
    root = logging.getLogger("pdpl")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)
    root.propagate = False
    return log_path


def get_logger(suffix: str = "") -> logging.Logger:
    return logging.getLogger("pdpl" + (f".{suffix}" if suffix else ""))

class PermanentError(Exception):
    """Failure that retrying cannot fix (e.g. request exceeds the context window)."""


class TransientError(Exception):
    """Temporary failure (OOM, I/O, network); retried with backoff."""

def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, PermanentError):
        return False
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    return True


def backoff_delay(attempt: int, cfg: dict) -> float:
    delay = cfg["base_delay_seconds"] * (cfg["backoff_factor"] ** (attempt - 1))
    delay = min(delay, cfg["max_delay_seconds"])
    jitter = cfg.get("jitter_seconds", 0) or 0
    return delay + (random.uniform(0, jitter) if jitter else 0.0)


def retry_call(fn: Callable, cfg: dict, sleep: Callable[[float], None] = time.sleep,
               on_retry: Callable[[int, BaseException, float], None] | None = None) -> Tuple[object, int]:
    max_attempts = int(cfg["max_attempts"])
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except BaseException as exc:      # noqa: BLE001 - classified below
            if not is_transient(exc) or attempt >= max_attempts:
                exc.attempts = attempt        # type: ignore[attr-defined]
                raise
            delay = backoff_delay(attempt, cfg)
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)


PACKAGES = ["torch", "transformers", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "langchain-community", "langchain-core",
            "faiss-cpu", "python-docx", "numpy", "PyYAML", "pandas", "openpyxl", "matplotlib"]


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": package_versions(), "cuda_available": None, "gpu": None}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:       # torch not installed
        pass
    return info


@dataclass(frozen=True)
class Rubric:
    items: List[str]
    text: str
    sha256: str

    @property
    def codes(self) -> List[str]:
        return [i.split(". ")[0] for i in self.items]

    def __len__(self) -> int:
        return len(self.items)


def load_rubric(path: str | Path) -> Rubric:
    raw = Path(path).read_text(encoding="utf-8")
    items = [ln.rstrip("\r") for ln in raw.split("\n") if ln.strip()]
    if not items:
        raise ValueError(f"Rubric file is empty: {path}")
    for it in items:
        if ". " not in it:
            raise ValueError(f"Rubric item without 'CODE. description' form: {it!r}")
    codes = [i.split(". ")[0] for i in items]
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate rubric codes")
    text = "\n".join(items)
    return Rubric(items=items, text=text, sha256=sha256_text(text))


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
    return t.render(rubric_text=rubric_text, section_title=title, section_content=content,
                    pdpl_text=pdpl_text or "")


def render_full_prompt(t: PromptTemplate, rubric_text: str, full_policy_text: str,
                       pdpl_context: str | None = None) -> str:
    return t.render(rubric_text=rubric_text, full_policy_text=full_policy_text,
                    pdpl_context=pdpl_context or "")


def parse_docx_to_sections(docx_path: str | Path) -> List[Dict]:
    from docx import Document
    doc = Document(str(docx_path))
    sections: List[Dict] = []
    current_section = {"title": "Introduction", "content": ""}
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if "Heading" in para.style.name:
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"title": text, "content": ""}
        else:
            current_section["content"] += " " + text
    if current_section["content"]:
        sections.append(current_section)
    return sections


def parse_docx_to_texts_metadatas(docx_path: str | Path) -> Tuple[List[str], List[Dict]]:
    from docx import Document as Docx
    doc = Docx(str(docx_path))
    texts: List[str] = []
    metadatas: List[Dict] = []
    current_section_content = ""
    current_section_title = "Introduction"
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name
        if "Heading" in style:
            if current_section_content:
                texts.append(current_section_content.strip())
                metadatas.append({"title": current_section_title, "content": str(docx_path)})
            current_section_title = text
            current_section_content = ""
        else:
            current_section_content += " " + text
    if current_section_content:
        texts.append(current_section_content.strip())
        metadatas.append({"title": current_section_title, "content": str(docx_path)})
    return texts, metadatas


def build_full_policy_text(sections: List[Dict]) -> str:
    """Notebook: "\\n".join(f"Section {i}: {title}\\n{content}" ...)."""
    return "\n".join(f"Section {i}: {s['title']}\n{s['content']}" for i, s in enumerate(sections, 1))


@dataclass
class PolicyData:
    path: Path
    sha256: str
    sections: List[Dict]
    full_text: str

    @property
    def n_sections(self) -> int:
        return len(self.sections)


def load_policy(path: str | Path) -> PolicyData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    sections = parse_docx_to_sections(path)
    if not sections:
        raise ValueError(f"No sections parsed from policy (empty or no text): {path}")
    return PolicyData(path=path, sha256=sha256_file(path), sections=sections,
                      full_text=build_full_policy_text(sections))


def load_labels(cfg, rubric):
    import openpyxl
    path = Path(cfg.ground_truth["xlsx_path"])
    path = path if path.is_absolute() else cfg.root / path
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = wb[cfg.ground_truth.get("sheet_name", "Expert Analysis")]
        rows = list(sheet.values)
        columns = {}
        for i, header in enumerate(rows[0]):
            match = re.match(r"^\s*([A-D]\d+)\s*\.", str(header or ""))
            if match:
                code = match.group(1)
                if code in columns:
                    raise ValueError(f"Duplicate ground-truth criterion {code}")
                columns[code] = i
        missing = set(rubric.codes) - set(columns)
        if missing:
            raise ValueError(f"Ground truth missing criteria: {sorted(missing)}")
        normalize = lambda name: " ".join(str(name).split()).casefold()
        lookup = {}
        for row in rows[1:]:
            if not row[0]:
                continue
            name = normalize(row[0])
            if name in lookup:
                raise ValueError(f"Duplicate application label row: {row[0]}")
            lookup[name] = row
        labels = {}
        aliases = cfg.ground_truth.get("application_aliases", {})
        for app in cfg.applications:
            name = normalize(aliases.get(app.id, app.name))
            if name not in lookup:
                raise ValueError(f"No expert label row for {app.name}; supply an explicit application_aliases entry")
            for code in rubric.codes:
                value = lookup[name][columns[code]]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
                    raise ValueError(f"Invalid/missing expert label: {app.name}/{code}: {value!r}")
                labels[app.id, code] = int(value)
        return labels, path
    finally:
        wb.close()
