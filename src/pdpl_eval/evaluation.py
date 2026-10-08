from __future__ import annotations
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .config import select
from .environment import environment_info
from .prompts import load_prompt
from .rubric import load_rubric
from .runner import can_reuse_result, experiment_fingerprint
from .storage import atomic_write_json, atomic_write_text, raw_path, load_json
from .utils import sha256_file, utc_now_iso

METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "kappa", "mcc", "false_positive_rate", "predicted_positive_rate", "true_positive_rate", "expert_positive_rate", "compliance_inflation")


def binary_metrics(y_true, y_pred):
    """Undefined metrics remain null, rather than being presented as measured zeros."""
    if len(y_true) != len(y_pred) or any(v not in (0, 1) for v in list(y_true) + list(y_pred)):
        raise ValueError("Evaluation requires equally sized binary label and prediction vectors")
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    n = len(y_true)
    div = lambda a, b: a / b if b else None
    recall, specificity = div(tp, tp + fn), div(tn, tn + fp)
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n ** 2 if n else 1
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=[0, 1])) if n and pe < 1 else None
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {"n": n, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": div(tp + tn, n),
            "precision": div(tp, tp + fp), "recall": recall, "specificity": specificity,
            "f1": div(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "kappa": kappa, "mcc": float(matthews_corrcoef(y_true, y_pred)) if denom else None,
            "false_positive_rate": div(fp, fp + tn), "predicted_positive_rate": div(tp + fp, n),
            "true_positive_rate": recall, "expert_positive_rate": div(tp + fn, n), "compliance_inflation": div(fp - fn, n)}


def load_labels(cfg, rubric):
    """Match exact criterion codes and normalized names; never infer absent labels as zero."""
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


def write_csv(path, rows):
    import io
    buffer = io.StringIO(newline="")
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(buffer, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        if row["complete"]:
            groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in groups.items():
        result = dict(zip(keys, key))
        result["n_complete_runs"] = len(group)
        result["at_least_five_runs_complete"] = len(group) >= 5
        for metric in METRICS + ("tp", "tn", "fp", "fn"):
            values = [r[metric] for r in group if r.get(metric) is not None and math.isfinite(r[metric])]
            result[metric + "_n_defined"] = len(values)
            result[metric + "_mean"] = mean(values) if values else None
            result[metric + "_sd"] = stdev(values) if len(values) > 1 else None
        output.append(result)
    return output


def analyze_results(cfg, allow_partial=False):
    """Score each complete repetition across all policies, then compute mean and sample SD."""
    rubric = load_rubric(cfg.rubric_path)
    labels, gt_path = load_labels(cfg, rubric)
    selection = select(cfg)
    output = cfg.results_dir / "analysis"
    cells, coverage, per_run, per_criterion = [], [], [], []
    provenance, section_cells = [], []
    for model in selection.models:
        for strategy in selection.strategies:
            template = load_prompt(strategy.name, strategy.prompt_path)
            for run in selection.run_numbers:
                unit_cells = []
                for app in selection.applications:
                    path = raw_path(cfg.results_dir, app.id, model.id, strategy.name, run)
                    state, reason = "missing", "result file missing"
                    try:
                        expected = experiment_fingerprint(cfg, app, model, strategy, rubric, template)
                        if path.exists():
                            record = load_json(path)
                            if can_reuse_result(path, expected, model.revision, cfg.retrieval.get("embedding_revision")):
                                state, reason = "success", ""
                                provenance.append({"path": str(path.relative_to(cfg.results_dir)), "sha256": sha256_file(path)})
                                for item, code in zip(rubric.items, rubric.codes):
                                    unit_cells.append({"model": model.id, "strategy": strategy.name, "run": run,
                                                       "application": app.id, "criterion": code, "criterion_text": item,
                                                       "expert": labels[app.id, code], "prediction": record["prediction"][item],
                                                       "correct": int(labels[app.id, code] == record["prediction"][item])})
                                if strategy.unit == "section":
                                    vectors = [x["parsed_vector"] for x in record["section_predictions"]]
                                    from .aggregation import aggregate
                                    rules = {"or": aggregate(vectors, "or"), "and": aggregate(vectors, "and"),
                                             "majority": aggregate(vectors, "majority"),
                                             "at_least_two": [int(sum(col) >= 2) for col in zip(*vectors)]}
                                    for rule, vector in rules.items():
                                        for code, value in zip(rubric.codes, vector):
                                            section_cells.append({"model": model.id, "strategy": strategy.name + "__" + rule,
                                                                  "run": run, "application": app.id, "criterion": code,
                                                                  "expert": labels[app.id, code], "prediction": value})
                            else:
                                state, reason = (record.get("status") if record.get("status") in ("failed", "parse_failed") else "stale_or_invalid"), "failed, invalid, unpinned or settings mismatch"
                    except Exception as exc:
                        state, reason = "invalid", f"{type(exc).__name__}: {exc}"
                    coverage.append({"model": model.id, "strategy": strategy.name, "run": run,
                                     "application": app.id, "state": state, "reason": reason})
                cells.extend(unit_cells)
                complete = len(unit_cells) == len(selection.applications) * len(rubric)
                identity = {"model": model.id, "strategy": strategy.name, "run": run, "complete": complete,
                            "expected_cells": len(selection.applications) * len(rubric)}
                if unit_cells:
                    per_run.append({**identity, **binary_metrics([r["expert"] for r in unit_cells], [r["prediction"] for r in unit_cells])})
                    for code in rubric.codes:
                        subset = [r for r in unit_cells if r["criterion"] == code]
                        per_criterion.append({**identity, "criterion": code,
                                              **binary_metrics([r["expert"] for r in subset], [r["prediction"] for r in subset])})
    main_summary = summarize(per_run, ["model", "strategy"])
    criterion_summary = summarize(per_criterion, ["model", "strategy", "criterion"])
    # Criterion macro average: average defined criterion metrics WITHIN each repetition first.
    macro_runs = []
    for row in per_run:
        subset = [r for r in per_criterion if all(r[k] == row[k] for k in ("model", "strategy", "run"))]
        macro = {k: row[k] for k in ("model", "strategy", "run", "complete")}
        for metric in METRICS:
            values = [r[metric] for r in subset if r[metric] is not None]
            macro[metric] = mean(values) if values else None
            macro[metric + "_n_criteria"] = len(values)
        macro_runs.append(macro)
    contrasts = []
    lookup = {(r["model"], r["strategy"], r["run"]): r for r in per_run if r["complete"]}
    pairs = [("section_rag", "section_norag"), ("full_rag", "full_norag"),
             ("full_rag", "section_rag"), ("full_norag", "section_norag")]
    for model in selection.models:
        for run in selection.run_numbers:
            for left, right in pairs:
                a, b = lookup.get((model.id, left, run)), lookup.get((model.id, right, run))
                if a and b:
                    contrasts.append({"model": model.id, "run": run, "contrast": left + " minus " + right,
                                      **{m: a[m] - b[m] if a[m] is not None and b[m] is not None else None for m in METRICS}})
    stability = []
    groups = defaultdict(list)
    for row in cells:
        groups[row["model"], row["strategy"], row["application"], row["criterion"]].append(row)
    for key, group in groups.items():
        positive = sum(r["prediction"] for r in group)
        stability.append({**dict(zip(("model", "strategy", "application", "criterion"), key)),
                          "n_runs": len(group), "positive_fraction": positive / len(group),
                          "all_runs_agree": len({r["prediction"] for r in group}) == 1,
                          "adjacent_run_flip_rate": adjacent_flip_rate(group)})
    for summary in main_summary:
        rates = [r["adjacent_run_flip_rate"] for r in stability if r["model"] == summary["model"] and r["strategy"] == summary["strategy"] and r["n_runs"] == cfg.runs and r["adjacent_run_flip_rate"] is not None]
        summary["run_to_run_flip_rate"] = mean(rates) if rates else None
    outputs = {"aligned_predictions.csv": cells, "coverage.csv": coverage, "per_run_metrics.csv": per_run,
               "main_table.csv": main_summary, "criterion_per_run.csv": per_criterion,
               "criterion_summary.csv": criterion_summary, "criterion_macro_per_run.csv": macro_runs,
               "criterion_macro_summary.csv": summarize(macro_runs, ["model", "strategy"]),
               "strategy_contrasts.csv": contrasts, "prediction_stability.csv": stability,
               "errors.csv": [r for r in cells if not r["correct"]]}
    outputs.update(additional_analysis(cfg, rubric, labels, cells, per_run, section_cells, output))
    for filename, rows in outputs.items():
        write_csv(output / filename, rows)
    complete = bool(coverage) and all(r["state"] == "success" for r in coverage) and len(selection.run_numbers) >= 5
    report = {"generated_at": utc_now_iso(), "analysis_complete": complete, "complete_matrix": all(r["state"] == "success" for r in coverage),
              "applications": len(selection.applications), "criteria": len(rubric), "runs": selection.run_numbers,
              "ground_truth_path": str(gt_path), "ground_truth_sha256": sha256_file(gt_path),
              "ground_truth_sheet": cfg.ground_truth.get("sheet_name", "Expert Analysis"),
              "inputs": provenance, "metric_convention": "positive label=1; undefined metrics=null; SD=sample SD across runs",
              "limitations": ["Expert labels must be confirmed adjudicated before publication",
                              "Cluster CIs resample applications, with the same sample used for all repetitions; conditional on supplied labels",
                              "Per-run exact item tests are descriptive because criteria within an application may be dependent",
                              "Keyword baseline requires independently reviewed patterns; evidence mitigation omitted by scope decision",
                              "Main summary includes only repetitions complete across all configured policies"],
              "keyword_baseline_configured": bool(cfg.raw.get("analysis", {}).get("keyword_patterns_path")),
              "independent_annotations_configured": bool(cfg.ground_truth.get("annotator_1_path") and cfg.ground_truth.get("annotator_2_path")),
              "analysis_environment": environment_info(),
              "files": list(outputs)}
    atomic_write_json(output / "analysis_manifest.json", report)
    if not complete and not allow_partial:
        raise ValueError(f"Matrix incomplete or fewer than five runs; diagnostic files saved to {output}. Use --allow-partial to inspect a pilot; do not publish it as the final table.")
    return report



def adjacent_flip_rate(rows):
    rows = sorted(rows, key=lambda r: r["run"])
    pairs = [(a, b) for a, b in zip(rows, rows[1:]) if b["run"] == a["run"] + 1]
    return sum(a["prediction"] != b["prediction"] for a, b in pairs) / len(pairs) if pairs else None


def cluster_intervals(rows, applications, samples=2000, seed=2026):
    """Resample whole applications; average repetition metrics inside each bootstrap sample."""
    import numpy as np
    if len(applications) < 2 or not rows:
        return []
    runs = sorted({r["run"] for r in rows})
    codes = sorted({r["criterion"] for r in rows})
    lookup = {(r["run"], r["application"], r["criterion"]): r for r in rows}
    if len(lookup) != len(runs) * len(applications) * len(codes):
        return []
    # Per-application confusion counts retain the cluster as the sampling unit.
    counts = np.zeros((len(applications), len(runs), 4), dtype=float)
    for i, app in enumerate(applications):
        for j, run in enumerate(runs):
            for code in codes:
                r = lookup[run, app, code]
                counts[i, j, 2 * r["expert"] + r["prediction"]] += 1
    draws = np.random.default_rng(seed).integers(0, len(applications), (samples, len(applications)))
    tn, fp, fn, tp = np.moveaxis(counts[draws].sum(axis=1), -1, 0)
    n = tn + fp + fn + tp
    def div(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
    recall, specificity = div(tp, tp + fn), div(tn, tn + fp)
    accuracy = div(tp + tn, n)
    pe = div((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn), n * n)
    values = {"accuracy": accuracy, "precision": div(tp, tp + fp), "recall": recall,
              "specificity": specificity, "f1": div(2 * tp, 2 * tp + fp + fn),
              "balanced_accuracy": (recall + specificity) / 2, "kappa": div(accuracy - pe, 1 - pe),
              "mcc": div(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
              "false_positive_rate": div(fp, fp + tn), "predicted_positive_rate": div(tp + fp, n),
              "true_positive_rate": recall, "expert_positive_rate": div(tp + fn, n), "compliance_inflation": div(fp - fn, n)}
    result = []
    for metric, value in values.items():
        # Undefined repetitions are not silently omitted from bootstrap estimands.
        valid = np.isfinite(value).all(axis=1)
        estimates = value[valid].mean(axis=1)
        low, high = np.quantile(estimates, [.025, .975]) if len(estimates) else (None, None)
        result.append({"metric": metric, "ci_low": low, "ci_high": high,
                       "n_bootstrap_defined": len(estimates), "n_bootstrap": samples,
                       "n_application_clusters": len(applications), "seed": seed})
    return result


def exact_test(a, b):
    from scipy.stats import binomtest
    return float(binomtest(a, a + b, .5).pvalue) if a + b else 1.0


def holm_adjust(rows):
    """Family = all tests in this output file; report raw and Holm-adjusted p values."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_value"])
    previous = 0
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(rows) - rank) * rows[i]["p_value"]))
        rows[i]["p_holm"] = previous


def additional_analysis(cfg, rubric, labels, cells, per_run, section_cells, output):
    settings = cfg.raw.get("analysis", {})
    samples, seed = int(settings.get("bootstrap_samples", 2000)), int(settings.get("bootstrap_seed", 2026))
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    applications = [a.id for a in cfg.applications]
    complete = {(r["model"], r["strategy"], r["run"]) for r in per_run if r["complete"]}
    valid_cells = [r for r in cells if (r["model"], r["strategy"], r["run"]) in complete]
    ci, asymmetry, paired = [], [], []
    groups = defaultdict(list)
    for row in valid_cells:
        groups[row["model"], row["strategy"]].append(row)
    for (model, strategy), rows in groups.items():
        ci.extend({"model": model, "strategy": strategy, **x} for x in cluster_intervals(rows, applications, samples, seed))
    for row in per_run:
        if row["complete"]:
            asymmetry.append({"model": row["model"], "strategy": row["strategy"], "run": row["run"],
                              "fp": row["fp"], "fn": row["fn"], "inflation": row["compliance_inflation"],
                              "test": "two-sided exact binomial on FP/FN; item independence assumption",
                              "p_value": exact_test(row["fp"], row["fn"])})
    lookup = {(r["model"], r["strategy"], r["run"], r["application"], r["criterion"]): r for r in valid_cells}
    pairs = [("section_rag", "section_norag"), ("full_rag", "full_norag"),
             ("full_rag", "section_rag"), ("full_norag", "section_norag")]
    for model in cfg.models:
        for left, right in pairs:
            for run in range(1, cfg.runs + 1):
                if (model.id, left, run) not in complete or (model.id, right, run) not in complete:
                    continue
                a_only = b_only = 0
                for app in applications:
                    for code in rubric.codes:
                        a, b = lookup[model.id, left, run, app, code], lookup[model.id, right, run, app, code]
                        ca, cb = a["prediction"] == a["expert"], b["prediction"] == b["expert"]
                        a_only += ca and not cb; b_only += cb and not ca
                paired.append({"model": model.id, "left": left, "right": right, "run": run,
                               "left_only_correct": a_only, "right_only_correct": b_only,
                               "accuracy_difference": (a_only - b_only) / len(labels),
                               "test": "exact McNemar; item independence assumption", "p_value": exact_test(a_only, b_only)})
    holm_adjust(asymmetry); holm_adjust(paired)
    # Alternative section rules use stored vectors, without changing primary OR outputs.
    aggregation_runs = []
    groups = defaultdict(list)
    for row in section_cells:
        groups[row["model"], row["strategy"], row["run"]].append(row)
    for (model, strategy, run), rows in groups.items():
        aggregation_runs.append({"model": model, "strategy": strategy, "run": run,
                                 "complete": len(rows) == len(labels),
                                 **binary_metrics([r["expert"] for r in rows], [r["prediction"] for r in rows])})
    baseline_cells = []
    for name, value in (("always_compliant", 1), ("always_not_disclosed", 0)):
        baseline_cells.extend({"model": "baseline", "strategy": name, "run": 1, "application": app,
                               "criterion": code, "expert": truth, "prediction": value}
                              for (app, code), truth in labels.items())
    pattern_path = settings.get("keyword_patterns_path")
    if pattern_path:
        import json
        from .policy import load_policy
        patterns = json.loads((cfg.root / pattern_path).read_text(encoding="utf-8"))
        if set(patterns) != set(rubric.codes) or any(not isinstance(v, list) or not v for v in patterns.values()):
            raise ValueError("Reviewed keyword baseline needs a nonempty regex list for every criterion")
        for app in cfg.applications:
            text = load_policy(app.policy_path).full_text
            for code in rubric.codes:
                value = int(any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns[code]))
                baseline_cells.append({"model": "baseline", "strategy": "reviewed_keyword", "run": 1,
                                       "application": app.id, "criterion": code, "expert": labels[app.id, code], "prediction": value})
    baselines, baseline_ci = [], []
    for name in sorted({r["strategy"] for r in baseline_cells}):
        rows = [r for r in baseline_cells if r["strategy"] == name]
        baselines.append({"baseline": name, **binary_metrics([r["expert"] for r in rows], [r["prediction"] for r in rows])})
        baseline_ci.extend({"baseline": name, **x} for x in cluster_intervals(rows, applications, samples, seed))
    human = []
    first, second = cfg.ground_truth.get("annotator_1_path"), cfg.ground_truth.get("annotator_2_path")
    if bool(first) != bool(second):
        raise ValueError("Supply both independent annotator workbooks, not one")
    if first and second:
        import copy
        c = copy.copy(cfg); c.ground_truth = dict(cfg.ground_truth)
        c.ground_truth["xlsx_path"] = first; one, _ = load_labels(c, rubric)
        c.ground_truth["xlsx_path"] = second; two, _ = load_labels(c, rubric)
        rows = [{"run": 1, "application": a, "criterion": code, "expert": one[a, code], "prediction": two[a, code]} for a, code in labels]
        for criterion in ["ALL"] + rubric.codes:
            subset = rows if criterion == "ALL" else [r for r in rows if r["criterion"] == criterion]
            metric = binary_metrics([r["expert"] for r in subset], [r["prediction"] for r in subset])
            interval = next(x for x in cluster_intervals(subset, applications, samples, seed) if x["metric"] == "kappa") if len(applications) > 1 else {}
            human.append({"criterion": criterion, "kappa": metric["kappa"], "n": metric["n"], **interval,
                          "annotator_1_sha256": sha256_file(cfg.root / first), "annotator_2_sha256": sha256_file(cfg.root / second)})
    human_ceiling = []
    if human:
        expert = next(r for r in human if r["criterion"] == "ALL")
        summaries = summarize(per_run, ["model", "strategy"])
        defined = [r for r in summaries if r.get("kappa_mean") is not None]
        if defined:
            best = max(defined, key=lambda r: r["kappa_mean"])
            human_ceiling.append({"model": best["model"], "strategy": best["strategy"],
                                  "model_expert_kappa_mean": best["kappa_mean"],
                                  "expert_expert_kappa": expert["kappa"],
                                  "expert_ci_low": expert.get("ci_low"), "expert_ci_high": expert.get("ci_high"),
                                  "selection": "descriptive best observed configuration; not a held-out selection"})
    # False-positive examples are candidates for manual inspection; no invented interpretation.
    candidates = [r for r in cells if r["expert"] == 0 and r["prediction"] == 1]
    draw_false_positive_heatmap(per_run, cells, rubric, output)
    return {"cluster_confidence_intervals.csv": ci, "fp_fn_asymmetry.csv": asymmetry,
            "paired_strategy_tests.csv": paired, "aggregation_per_run.csv": aggregation_runs,
            "aggregation_summary.csv": summarize(aggregation_runs, ["model", "strategy"]),
            "baseline_predictions.csv": baseline_cells, "baseline_metrics.csv": baselines,
            "baseline_confidence_intervals.csv": baseline_ci, "expert_agreement.csv": human, "human_ceiling_comparison.csv": human_ceiling,
            "false_positive_cases.csv": candidates}


def draw_false_positive_heatmap(per_run, cells, rubric, output):
    if not cells:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    complete = {(r["model"], r["strategy"], r["run"]) for r in per_run if r["complete"]}
    groups = sorted({(r["model"], r["strategy"]) for r in cells})
    matrix = np.full((len(groups), len(rubric)), np.nan)
    for i, (model, strategy) in enumerate(groups):
        for j, code in enumerate(rubric.codes):
            rates = []
            for run in range(1, max(r["run"] for r in cells) + 1):
                if (model, strategy, run) not in complete:
                    continue
                rows = [r for r in cells if (r["model"], r["strategy"], r["run"], r["criterion"]) == (model, strategy, run, code) and r["expert"] == 0]
                if rows:
                    rates.append(sum(r["prediction"] for r in rows) / len(rows))
            if rates:
                matrix[i, j] = mean(rates)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(3, len(groups) * .4)))
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(rubric)), rubric.codes)
    ax.set_yticks(range(len(groups)), [m + " / " + s for m, s in groups])
    ax.set_title("False-positive compliance rate (mean across complete repetitions)")
    fig.colorbar(image, ax=ax, label="FP / (FP + TN); blank = undefined or incomplete")
    fig.tight_layout()
    fig.savefig(output / "false_positive_heatmap.png", dpi=200)
    fig.savefig(output / "false_positive_heatmap.pdf")
    plt.close(fig)
