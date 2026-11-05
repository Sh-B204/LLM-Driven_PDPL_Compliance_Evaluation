import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
try:
    from scipy.stats import binomtest           
except ImportError:
    raise SystemExit("ERROR: scipy is required for the exact error-asymmetry test. Install it with: python -m pip install scipy")

RULES = {"or": lambda v: (v.max(0) >= 1), "and": lambda v: (v.min(0) >= 1),
         "majority": lambda v: (2 * v.sum(0) > len(v)), "at_least_two": lambda v: (v.sum(0) >= 2)}
MODELS = ["allam-7b", "llama-3.1-8b", "mistral-7b", "qwen-2.5-7b"]
STRATS = ["section_norag", "section_rag", "full_norag", "full_rag"]
FIRST_RUN = 1                                       # the error-asymmetry test uses this repetition only (never pooled over runs)
ASYM_TEST = "two-sided exact McNemar test (exact binomial test of FP among FP+FN errors, p0 = 0.5); repetition 1 only"
ASYM_NOTE = ("Unadjusted, exploratory p-value (no correction across configurations). Assumes independent decision pairs, but criteria "
             "within a policy may be related, so it is NOT cluster-adjusted; using a single run does not remove within-policy dependence.")

def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def load(repo, results="results", config="configs/experiment.yaml"):
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from pdpl.compat import classify_records
    repo = Path(repo); results_dir = (repo / results) if not Path(results).is_absolute() else Path(results)
    labels = classify_records(results_dir, repo, config)
    codes = [l.split(". ")[0] for l in (repo / "rubric/rubric_items.txt").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    gt = pd.read_excel(repo / "data/ground_truth/PDPL_Results.xlsx", sheet_name="Expert Analysis")
    gt = gt.rename(columns={gt.columns[0]: "app"})
    cols = {c: re.match(r"\s*([A-D]\d+)\.", str(c)).group(1) for c in gt.columns[1:] if re.match(r"\s*([A-D]\d+)\.", str(c))}
    gt = gt.rename(columns=cols).dropna(subset=["app"]); gt["key"] = gt["app"].map(norm)
    if gt["key"].duplicated().any() or gt[codes].isna().any().any() or not gt[codes].isin([0, 1]).all().all():
        raise ValueError("Expert labels must have unique applications and complete binary criterion labels")
    expert = {r.key: np.array([r[c] for c in codes], float) for _, r in gt.iterrows()}
    rows, fails = [], []
    excluded = []
    for f in sorted(results_dir.glob("raw/*/*/*/run_*.json")):
        d = json.loads(f.read_text(encoding="utf-8-sig")); k = norm(d["application"]["name"])
        if labels[f] not in ("archived_protocol_1.1.0", "protocol_2.x_verified"):
            excluded.append({"path": str(f), "label": labels[f]}); continue
        base = dict(provenance=labels[f], app=d["application"]["name"], key=k, model=d["model"]["id"], strategy=d["strategy"], run=d["run"], unit=d["unit"], rag=d["rag"])
        if d["status"] != "success" or d.get("prediction") is None:
            fails.append({**base, "status": d["status"], "reason": (d.get("error") or {}).get("type"),
                          "prompt_tokens": d.get("token_lengths", {}).get("prompt_tokens")}); continue
        if d["unit"] == "section":
            secs = np.array([s["parsed_vector"] for s in d["section_predictions"]], int)
            for r, fn in RULES.items():
                rows.append({**base, "rule": r, "vec": fn(secs).astype(int), "n_sections": len(secs), "secs": secs})
        else:
            rows.append({**base, "rule": "full", "vec": np.array([d["prediction"][i] for i in d["rubric"]["items"]], int), "n_sections": 1, "secs": None})
    return codes, expert, pd.DataFrame(rows), pd.DataFrame(fails), pd.DataFrame(excluded), pd.Series(labels).map(str).value_counts()

def cell_metrics(y, p):
    tp = ((y == 1) & (p == 1)).sum(); tn = ((y == 0) & (p == 0)).sum(); fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    d = lambda a, b: a / b if b else np.nan
    f1p = d(2 * tp, 2 * tp + fp + fn); f1n = d(2 * tn, 2 * tn + fn + fp)
    rec, spec = d(tp, tp + fn), d(tn, tn + fp)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    n = len(y); po = (tp + tn) / n if n else np.nan; pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2 if n else np.nan
    return dict(accuracy=d(tp + tn, n), precision=d(tp, tp + fp), recall=rec, specificity=spec,
                balanced_accuracy=(rec + spec) / 2,                       # mean of recall on the two classes (NaN if a class is absent)
                mcc=(tp * tn - fp * fn) / den if den else np.nan,         # Matthews correlation; NaN when a margin is empty
                kappa=(po - pe) / (1 - pe) if n and pe != 1 else np.nan,  # Cohen's kappa vs chance agreement with the expert labels
                f1_positive=f1p, f1_class_macro=np.nanmean([f1p, f1n]), fp=fp, fn=fn, n=n)

def confusion_counts(y, p):
    y = np.asarray(y).astype(int).ravel(); p = np.asarray(p).astype(int).ravel()
    return dict(tn=int(((y == 0) & (p == 0)).sum()), fp=int(((y == 0) & (p == 1)).sum()),
                fn=int(((y == 1) & (p == 0)).sum()), tp=int(((y == 1) & (p == 1)).sum()))

def error_asymmetry(fp, fn, conf=0.95):
    fp, fn = int(fp), int(fn); n = fp + fn
    out = dict(fp=fp, fn=fn, total_errors=n, fp_share=np.nan, fp_share_ci_lo=np.nan, fp_share_ci_hi=np.nan, p_value=1.0)
    if n:
        r = binomtest(fp, n, 0.5, alternative="two-sided"); ci = r.proportion_ci(confidence_level=conf, method="exact")
        out.update(fp_share=fp / n, fp_share_ci_lo=float(ci.low), fp_share_ci_hi=float(ci.high), p_value=float(r.pvalue))
    return out

def run_metrics(sub, expert, with_baseline=False):
    Y = np.stack([expert[k] for k in sub.key]); P = np.stack(sub.vec.values)
    m = cell_metrics(Y.ravel(), P.ravel())
    pc = [cell_metrics(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    m["criterion_macro_accuracy"] = np.nanmean([c["accuracy"] for c in pc])
    m["criterion_macro_f1_positive"] = np.nanmean([c["f1_positive"] for c in pc])
    pct_p, pct_y = P.mean(1) * 100, Y.mean(1) * 100
    m["pct_mae"] = np.abs(pct_p - pct_y).mean(); m["pct_signed_bias"] = (pct_p - pct_y).mean()
    return m

def summarize(df, expert, keys, label):
    out = []
    for g, d in df.groupby(keys):
        per = [{**run_metrics(r, expert), "run": run} for run, r in d.groupby("run")]
        t = pd.DataFrame(per); row = dict(zip(keys, g if isinstance(g, tuple) else (g,)))
        row.update(n_policies=d.key.nunique(), n_runs=len(t), n_cells=int(t.n.iloc[0]))
        for c in t.columns.drop(["run", "n", "fp", "fn"]):
            row[c + "_mean"] = t[c].mean(); row[c + "_sd"] = t[c].std(ddof=1) if len(t) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out).assign(subset=label)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--out", default="evaluation"); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--results", default="results", help="results folder under --repo: results (archived study) or results_v2 (new inference)")
    ap.add_argument("--config", default="configs/experiment.yaml", help="config used to recompute fingerprints of protocol-2 records")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.boot < 100: ap.error("--boot must be at least 100")
    codes, expert, df, fails, excluded, prov = load(a.repo, a.results, a.config)
    prov.rename("n_records").rename_axis("provenance_label").to_csv(out / "provenance_summary.csv"); excluded.to_csv(out / "provenance_exclusions.csv", index=False)
    print("provenance:", prov.to_dict())
    if df.empty and fails.empty: raise SystemExit("ERROR: no record passed provenance validation; nothing to score")
    if df.empty: raise SystemExit("ERROR: no successful predictions to score")
    if (prov.index.isin(["archived_protocol_1.1.0", "protocol_2.x_verified"])).sum() > 1:
        raise SystemExit("ERROR: evaluate archived and new inference protocols separately")
    NR = int(df.run.nunique())                      # repetitions present (5 for the baseline); "complete" = all NR runs present
    primary = df[(df.rule == "or") | (df.rule == "full")].copy()   # section strategies use OR as the notebook did
    cfg = ["model", "strategy"]
    total = df.drop_duplicates(["model", "strategy", "key", "run"]).groupby(cfg).size()
    nfail = fails.groupby(cfg).size() if len(fails) else pd.Series(0, index=total.index)
    cov = pd.DataFrame({"successful_records": total}).join(nfail.rename("failed_records")).fillna(0)
    cov["policies_complete_5_runs"] = primary.groupby(cfg + ["key"]).run.nunique().eq(NR).groupby(cfg).sum()
    cov.to_csv(out / "coverage.csv")
    if len(fails): fails.to_csv(out / "failed_records.csv", index=False)
    complete = primary.groupby(cfg + ["key"]).run.nunique(); complete = complete[complete == NR].reset_index()
    if complete.empty: raise SystemExit("ERROR: no policy has all repetitions for any configuration")
    matched_keys = set.intersection(*[set(g.key) for _, g in complete.groupby(cfg)])
    avail = primary[primary.set_index(cfg + ["key"]).index.isin(complete.set_index(cfg + ["key"]).index)]
    grand_a = summarize(avail, expert, cfg, "available"); grand_m = summarize(avail[avail.key.isin(matched_keys)], expert, cfg, "matched")
    grand = pd.concat([grand_a, grand_m]); grand.to_csv(out / "grand_table.csv", index=False)
    base = []
    for lab, keys in [("available", list(expert.keys() & set(primary.key))), ("matched", sorted(matched_keys))]:
        Y = np.stack([expert[k] for k in keys]).ravel()
        for n, v in [("all-disclosed(1)", 1), ("all-missing(0)", 0)]:
            base.append(dict(subset=lab, baseline=n, n_policies=len(keys), **cell_metrics(Y, np.full_like(Y, v))))
    pd.DataFrame(base).to_csv(out / "constant_baselines.csv", index=False)
    cb = []
    for (m, s), d in avail.groupby(cfg):
        keys = sorted(d.key.unique()); Y = np.stack([expert[k] for k in keys]).ravel()
        for name, value in [("all-disclosed(1)", 1), ("all-missing(0)", 0)]:
            cb.append(dict(model=m, strategy=s, baseline=name, n_policies=len(keys), **cell_metrics(Y, np.full_like(Y, value))))
    pd.DataFrame(cb).to_csv(out / "baselines_by_configuration.csv", index=False)
    sec = df[df.unit == "section"]
    agg = summarize(sec, expert, ["model", "strategy", "rule"], "all_available_section_records")
    agg.to_csv(out / "aggregation_rules.csv", index=False)

    crit = []
    for (m, s), d in avail.groupby(cfg):
        Y = np.stack([expert[k] for k in d.key]); P = np.stack(d.vec.values)
        for j, c in enumerate(codes):
            crit.append(dict(model=m, strategy=s, criterion=c, n_policies=d.key.nunique(), agreement=(Y[:, j] == P[:, j]).mean(),
                             prevalence_baseline=Y[:, j].mean(),
                             fp_rate=((Y[:, j] == 0) & (P[:, j] == 1)).sum() / (Y[:, j] == 0).sum() if (Y[:, j] == 0).any() else np.nan,
                             fn_rate=((Y[:, j] == 1) & (P[:, j] == 0)).sum() / (Y[:, j] == 1).sum() if (Y[:, j] == 1).any() else np.nan,
                             expert_pos=Y[:, j].mean(), pred_pos=P[:, j].mean()))
    crit = pd.DataFrame(crit); crit.to_csv(out / "criterion_level.csv", index=False)

    pcm = []
    for (m, s), d in avail.groupby(cfg):
        Y = np.stack([expert[k] for k in d.key]); P = np.stack(d.vec.values); cc = confusion_counts(Y, P)
        pcm.append(dict(model=m, strategy=s, rule=d.rule.iloc[0], n_policies=d.key.nunique(), n_runs=d.run.nunique(), n_decisions=int(Y.size), **cc,
                        expert_negative=cc["tn"] + cc["fp"], expert_positive=cc["fn"] + cc["tp"]))
    pcm = pd.DataFrame(pcm); pcm.to_csv(out / "pooled_confusion_matrices.csv", index=False)
    if (pcm.tn + pcm.fp + pcm.fn + pcm.tp != pcm.n_decisions).any() or (pcm.n_decisions != pcm.n_policies * pcm.n_runs * len(codes)).any():
        raise SystemExit("ERROR: pooled confusion counts do not reconcile with policies x runs x criteria")
    if FIRST_RUN not in set(avail.run): raise SystemExit(f"ERROR: repetition {FIRST_RUN} is not present; the error-asymmetry test needs it")
    asy = []
    for (m, s), d in avail.groupby(cfg):
        d1 = d[d.run == FIRST_RUN]                                        # same policies as the Table 5 row (complete in all repetitions)
        Y = np.stack([expert[k] for k in d1.key]); P = np.stack(d1.vec.values); cc = confusion_counts(Y, P)
        asy.append(dict(model=m, strategy=s, run=FIRST_RUN, n_policies=d1.key.nunique(), n_decisions=int(Y.size), **error_asymmetry(cc["fp"], cc["fn"]),
                        test=ASYM_TEST, note=ASYM_NOTE))
    asy = pd.DataFrame(asy); asy.to_csv(out / "error_asymmetry_run1.csv", index=False)
    chk = asy.merge(grand_a[cfg + ["n_policies"]], on=cfg, suffixes=("", "_table"))
    if len(chk) != len(asy) or (chk.n_policies != chk.n_policies_table).any(): raise SystemExit("ERROR: run-1 asymmetry coverage differs from the available-policy table")

    st = []
    for (m, s), d in avail.groupby(cfg):
        ag = []
        for k, g in d.groupby("key"):
            V = np.stack(g.sort_values("run").vec.values)
            ag += [(V[i] == V[j]).mean() for i in range(len(V)) for j in range(i + 1, len(V))]
        st.append(dict(model=m, strategy=s, mean_pairwise_run_agreement=np.mean(ag)))
    pd.DataFrame(st).to_csv(out / "repetition_agreement.csv", index=False)

    ap_rows = []
    for (m, s, k, r), d in avail.groupby(cfg + ["key", "run"]):
        ap_rows.append(dict(app=d.app.iloc[0], key=k, model=m, strategy=s, run=r, pred_pct=d.vec.iloc[0].mean() * 100, expert_pct=expert[k].mean() * 100))
    apdf = pd.DataFrame(ap_rows); apdf["error_pp"] = apdf.pred_pct - apdf.expert_pct
    apm = apdf.groupby(["app", "model", "strategy"]).agg(pred_pct_mean=("pred_pct", "mean"), pred_pct_sd=("pred_pct", lambda x: x.std(ddof=1)), expert_pct=("expert_pct", "first"), error_pp_mean=("error_pp", "mean")).reset_index()
    apm.to_csv(out / "application_percentages.csv", index=False)

    rng = np.random.default_rng(2026); pr = []
    for m in sorted(avail.model.unique()):
        for unit in ("section", "full"):
            sc = {}
            for rag in ("norag", "rag"):
                d = avail[(avail.model == m) & (avail.strategy == f"{unit}_{rag}")]
                sc[rag] = pd.Series({k: np.mean([(v == expert[k]).mean() for v in g.vec]) for k, g in d.groupby("key")})
            ks = sorted(set(sc["rag"].index) & set(sc["norag"].index))
            if not ks: continue
            diff = np.array([sc["rag"][k] - sc["norag"][k] for k in ks])
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(a.boot)]
            pr.append(dict(model=m, unit=unit, n_policies=len(ks), mean_diff_accuracy=diff.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    pr = pd.DataFrame(pr); pr.to_csv(out / "paired_rag_effect.csv", index=False)

    ci = []
    for (m, s), d in avail.groupby(cfg):
        keys = sorted(d.key.unique()); runs = sorted(d.run.unique())
        Y = np.stack([expert[k] for k in keys])
        P = np.stack([np.stack(d[d.run == r].set_index("key").loc[keys, "vec"].values) for r in runs])
        bs = []
        for _ in range(a.boot):
            ix = rng.integers(0, len(keys), len(keys)); yy = Y[ix].ravel()
            mm = [cell_metrics(yy, P[j, ix].ravel()) for j in range(len(runs))]
            bs.append([np.mean([x["accuracy"] for x in mm]), np.nanmean([x["f1_positive"] for x in mm])])
        bs = np.array(bs)
        ci.append(dict(model=m, strategy=s, n_policies=len(keys), acc_lo=np.percentile(bs[:, 0], 2.5), acc_hi=np.percentile(bs[:, 0], 97.5), f1_lo=np.nanpercentile(bs[:, 1], 2.5), f1_hi=np.nanpercentile(bs[:, 1], 97.5)))
    pd.DataFrame(ci).to_csv(out / "policy_cluster_ci.csv", index=False)

    with pd.ExcelWriter(out / "results_summary.xlsx") as xw:
        grand.to_excel(xw, sheet_name="grand_table", index=False); cov.reset_index().to_excel(xw, sheet_name="coverage", index=False)
        agg.to_excel(xw, sheet_name="aggregation_rules", index=False); crit.to_excel(xw, sheet_name="criterion_level", index=False)
        apm.to_excel(xw, sheet_name="application_pct", index=False); pr.to_excel(xw, sheet_name="paired_rag", index=False)
        pcm.to_excel(xw, sheet_name="pooled_confusion", index=False); asy.to_excel(xw, sheet_name="error_asymmetry_run1", index=False)
        (fails if len(fails) else pd.DataFrame({"note": ["no failures"]})).to_excel(xw, sheet_name="failed_records", index=False)
    final = pd.DataFrame([{**{c: r[c] for c in ("app", "model", "strategy", "run", "rule")}, **dict(zip(codes, r.vec))} for _, r in df.iterrows()])
    final.to_csv(out / "final_vectors_15_items.csv", index=False)
    secrows = [{"app": r.app, "model": r.model, "strategy": r.strategy, "run": r.run, "section": i + 1, **dict(zip(codes, v))}
               for _, r in df[df.rule == "or"].iterrows() if r.secs is not None for i, v in enumerate(r.secs)]
    pd.DataFrame(secrows).to_csv(out / "section_vectors.csv", index=False)
    print("matched policies:", len(matched_keys), "| records scored:", len(df.drop_duplicates(["model","strategy","key","run"])), "| failures:", len(fails))

if __name__ == "__main__": main()
