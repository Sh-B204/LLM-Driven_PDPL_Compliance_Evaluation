from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

STRATEGY_NAMES = ("section_rag", "full_rag", "section_norag", "full_norag")
STRATEGY_NOTEBOOK_LABELS = {
    "section_rag": "Section-by-Section (OR)",
    "full_rag": "Full-Policy+Law (Experiment 1)",
    "section_norag": "(new) Section-by-Section without RAG",
    "full_norag": "Full-Policy",
}
MODEL_TYPES = ("hf_local",)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Application:
    name: str
    id: str
    policy_path: Path


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    type: str
    hf_id: str
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    tokenizer_use_fast: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "float16"
    device_map: str = "auto"
    prompt_template: Optional[str] = None
    context_limit_tokens: Optional[int] = None


@dataclass(frozen=True)
class StrategySpec:
    name: str
    enabled: bool
    unit: str
    rag: bool
    prompt_path: Path
    aggregation: Optional[str] = None


@dataclass
class Config:
    root: Path
    runs: int
    applications: List[Application]
    models: List[ModelSpec]
    strategies: Dict[str, StrategySpec]
    rubric_path: Path
    retrieval: dict
    generation: dict
    retry: dict
    output: dict
    ground_truth: dict
    source_file: Path
    raw: dict = field(default_factory=dict)

    @property
    def results_dir(self) -> Path:
        return self.root / self.output["results_dir"]

    def with_results_dir(self, path: str | Path) -> "Config":
        """Copy of the config writing to another results directory (used by the test scripts)."""
        import copy
        c = copy.copy(self)
        c.output = dict(self.output)
        p = Path(path)
        c.output["results_dir"] = str(p if p.is_absolute() else p)
        return c


def _resolve(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file is empty or not a mapping: {path}")
    return data


def _require(d: dict, key: str, where: str):
    if key not in d or d[key] is None:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return d[key]


def load_config(experiment_file: str | Path = "configs/experiment.yaml", root: Path | None = None) -> Config:
    root = Path(root) if root else REPO_ROOT
    exp_path = _resolve(root, experiment_file)
    raw = _load_yaml(exp_path)

    runs = _require(raw, "runs", exp_path)
    if not isinstance(runs, int) or runs < 1:
        raise ConfigError("'runs' must be a positive integer")

    apps_file = _resolve(root, _require(raw, "applications_file", exp_path))
    apps_raw = _require(_load_yaml(apps_file), "applications", apps_file)
    applications: List[Application] = []
    for a in apps_raw:
        for k in ("name", "id", "policy_path"):
            _require(a, k, f"{apps_file} (application entry {a})")
        if not ID_RE.match(a["id"]):
            raise ConfigError(f"Invalid application id '{a['id']}' (use lowercase letters, digits, _ . -)")
        applications.append(Application(a["name"], a["id"], _resolve(root, a["policy_path"])))
    _check_unique([a.id for a in applications], "application ids")
    _check_unique([a.name for a in applications], "application names")

    models_file = _resolve(root, _require(raw, "models_file", exp_path))
    models_raw = _require(_load_yaml(models_file), "models", models_file)
    models: List[ModelSpec] = []
    for m in models_raw:
        for k in ("id", "label", "type", "hf_id"):
            _require(m, k, f"{models_file} (model entry {m})")
        if not ID_RE.match(m["id"]):
            raise ConfigError(f"Invalid model id '{m['id']}'")
        if m["type"] not in MODEL_TYPES:
            raise ConfigError(f"Model '{m['id']}': type must be one of {MODEL_TYPES}")
        known = {f for f in ModelSpec.__dataclass_fields__}
        unknown = set(m) - known
        if unknown:
            raise ConfigError(f"Model '{m['id']}': unknown keys {sorted(unknown)}")
        models.append(ModelSpec(**m))
    _check_unique([m.id for m in models], "model ids")

    strategies: Dict[str, StrategySpec] = {}
    for name, s in _require(raw, "strategies", exp_path).items():
        if name not in STRATEGY_NAMES:
            raise ConfigError(f"Unknown strategy '{name}'. Allowed: {STRATEGY_NAMES}")
        unit = _require(s, "unit", f"strategy {name}")
        if unit not in ("section", "full"):
            raise ConfigError(f"Strategy {name}: unit must be 'section' or 'full'")
        if name.startswith("section") != (unit == "section") or name.endswith("_rag") != bool(s.get("rag")):
            raise ConfigError(f"Strategy {name}: 'unit'/'rag' settings are inconsistent with the strategy name")
        agg = s.get("aggregation")
        if unit == "section" and agg not in ("or",):
            raise ConfigError(f"Strategy {name}: aggregation must be 'or' (the notebook's rule)")
        strategies[name] = StrategySpec(name, bool(s.get("enabled", True)), unit, bool(s["rag"]),
                                        _resolve(root, _require(s, "prompt_path", f"strategy {name}")), agg)

    for key in ("retrieval", "generation", "retry", "output"):
        _require(raw, key, exp_path)
    for key in ("corpus_path", "embedding_model", "embedding_max_length", "top_k_section", "top_k_full",
                "section_context_separator", "full_context_separator"):
        _require(raw["retrieval"], key, f"{exp_path} retrieval")
    for key in ("max_new_tokens", "temperature", "top_p", "repetition_penalty", "do_sample"):
        _require(raw["generation"], key, f"{exp_path} generation")
    for key in ("max_attempts", "base_delay_seconds", "backoff_factor", "max_delay_seconds"):
        _require(raw["retry"], key, f"{exp_path} retry")
    _require(raw["output"], "results_dir", f"{exp_path} output")

    return Config(
        root=root, runs=runs, applications=applications, models=models, strategies=strategies,
        rubric_path=_resolve(root, _require(raw, "rubric_path", exp_path)),
        retrieval=raw["retrieval"], generation=raw["generation"], retry=raw["retry"],
        output=raw["output"], ground_truth=raw.get("ground_truth", {}) or {},
        source_file=exp_path, raw=raw,
    )


def _check_unique(values: List[str], what: str):
    dup = {v for v in values if values.count(v) > 1}
    if dup:
        raise ConfigError(f"Duplicate {what}: {sorted(dup)}")


@dataclass
class Selection:
    applications: List[Application]
    models: List[ModelSpec]
    strategies: List[StrategySpec]
    run_numbers: List[int]


def _match(items, wanted: Optional[List[str]], keys, what: str):
    if not wanted:
        return list(items)
    wanted_l = [w.lower() for w in wanted]
    chosen = []
    for w in wanted_l:
        hit = [i for i in items if w in [str(getattr(i, k)).lower() for k in keys]]
        if not hit:
            avail = [getattr(i, keys[0]) for i in items]
            raise ConfigError(f"Unknown {what} '{w}'. Available: {avail}")
        for h in hit:
            if h not in chosen:
                chosen.append(h)
    return chosen


def select(cfg: Config, applications=None, models=None, strategies=None,
           runs: Optional[int] = None, run_numbers: Optional[List[int]] = None) -> Selection:
    """Apply CLI filters. 'runs' keeps run numbers 1..runs; 'run_numbers' picks explicit numbers."""
    apps = _match(cfg.applications, applications, ("id", "name"), "application")
    mods = _match(cfg.models, models, ("id", "label"), "model")
    enabled = [s for s in cfg.strategies.values() if s.enabled]
    if strategies:
        for s in strategies:
            if s not in STRATEGY_NAMES:
                raise ConfigError(f"Unknown strategy '{s}'. Allowed: {STRATEGY_NAMES}")
            if s not in cfg.strategies:
                raise ConfigError(f"Strategy '{s}' is not defined in the configuration")
        strats = [cfg.strategies[s] for s in strategies]
    else:
        strats = enabled
    if run_numbers:
        bad = [r for r in run_numbers if r < 1]
        if bad:
            raise ConfigError("run numbers must be >= 1")
        nums = sorted(set(run_numbers))
    else:
        n = runs if runs is not None else cfg.runs
        if n < 1:
            raise ConfigError("--runs must be >= 1")
        nums = list(range(1, n + 1))
    return Selection(apps, mods, strats, nums)
