from __future__ import annotations
import json, sys
from pathlib import Path

def _protocol_v1_modules():
    import importlib
    return {m: importlib.import_module(f".protocol_v1.{m}", package=__package__)
            for m in ("config", "runner", "prompts", "rubric", "validation")}


def classify_archived(results_dir, repo_root, config="configs/experiment.yaml"):
    L = _protocol_v1_modules(); repo_root = Path(repo_root)
    cfg = L["config"].load_config(str(repo_root / config), root=repo_root); rub = L["rubric"].load_rubric(cfg.rubric_path)
    apps = {a.id: a for a in cfg.applications}; models = {m.id: m for m in cfg.models}; specs = cfg.strategies; labels = {}
    for f in sorted(Path(results_dir).glob("raw/*/*/*/run_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8")); problems = L["validation"].validate_record(rec, f)
        if rec.get("pipeline_version") != "1.1.0": labels[f] = "not_archived_1.1.0"; continue
        sp = specs[rec["strategy"]]; tpl = L["prompts"].load_prompt(sp.name, sp.prompt_path)
        fp = L["runner"].experiment_fingerprint(cfg, apps[rec["application"]["id"]], models[rec["model"]["id"]], sp, rub, tpl)
        labels[f] = "archived_protocol_1.1.0" if (not problems and fp == rec.get("experiment_fingerprint")) else "unverified"
    return labels

def classify_records(results_dir, repo_root, config="configs/experiment.yaml"):
    import copy
    repo_root, labels = Path(repo_root), {}
    files = sorted(Path(results_dir).glob("raw/*/*/*/run_*.json"))
    versions = {f: json.loads(f.read_text(encoding="utf-8")).get("pipeline_version", "") for f in files}
    if any(v == "1.1.0" for v in versions.values()):
        arch = classify_archived(results_dir, repo_root, config)
        labels.update({f: l for f, l in arch.items() if versions.get(f) == "1.1.0"})
    new = [f for f, v in versions.items() if v.startswith("2.")]
    if new:
        from .config import load_config
        from .inputs import load_prompt, load_rubric
        from .records import validate_record
        from .runner import experiment_fingerprint
        cfg = load_config(str(repo_root / config), root=repo_root); rub = load_rubric(cfg.rubric_path)
        apps = {a.id: a for a in cfg.applications}; models = {m.id: m for m in cfg.models}
        for f in new:
            rec = json.loads(f.read_text(encoding="utf-8")); probs = validate_record(rec, f)
            try:
                c2 = copy.copy(cfg); c2.generation = dict(rec["parameters"]["generation"])
                sp = cfg.strategies[rec["strategy"]]
                fp = experiment_fingerprint(c2, apps[rec["application"]["id"]], models[rec["model"]["id"]], sp, rub, load_prompt(sp.name, sp.prompt_path))
            except Exception as e:
                labels[f] = f"unverified: cannot recompute fingerprint ({type(e).__name__}: {e})"; continue
            labels[f] = ("protocol_2.x_verified" if not probs and fp == rec.get("experiment_fingerprint")
                         else "unverified: " + ("; ".join(probs) if probs else "fingerprint mismatch (code/config/inputs differ from the record)"))
    for f, v in versions.items():
        labels.setdefault(f, f"unverified: unknown pipeline_version '{v}'")
    return labels

if __name__ == "__main__":
    import collections
    lab = classify_archived(sys.argv[1], sys.argv[2]); print(collections.Counter(lab.values()))
