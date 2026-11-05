import argparse, glob, json, re, hashlib
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
AR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"); LA = re.compile(r"[A-Za-z]")

def extract_like_pipeline(doc):
    secs, cur = [], {"title": "Introduction", "content": ""}
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t: continue
        if "Heading" in p.style.name:
            if cur["content"]: secs.append(cur)
            cur = {"title": t, "content": ""}
        else: cur["content"] += " " + t
    if cur["content"]: secs.append(cur)
    return secs

def text_of(el): return "".join(t.text or "" for t in el.iter(W + "t"))

def audit_docx(doc, secs):
    body = doc.element.body
    extracted = sum(len(p.text.strip()) for p in doc.paragraphs)
    tables = list(body.iter(W + "tbl")); table_chars = sum(len(text_of(t).strip()) for t in tables if t.getparent() is body or True) if tables else 0
    # text boxes: Word stores each twice (mc:Choice + mc:Fallback); keep only non-fallback to avoid double counting
    tb = [e for e in body.iter(W + "txbxContent") if not any(a.tag.endswith("Fallback") for a in e.iterancestors())]
    tb_chars = sum(len(text_of(e).strip()) for e in tb)
    hf = 0
    for s in doc.sections:
        for part in (s.header, s.footer):
            hf += sum(len(p.text.strip()) for p in part.paragraphs)
    sdt = sum(len(text_of(e).strip()) for e in body.iter(W + "sdtContent") if e.getparent().getparent() is body)
    heads = sum("Heading" in p.style.name for p in doc.paragraphs)
    pre = bool(secs and secs[0]["title"] == "Introduction")
    return dict(paragraph_chars_extracted=extracted, n_tables=len(tables), table_chars_not_extracted=table_chars,
                n_textboxes=len(tb), textbox_chars_not_extracted=tb_chars, header_footer_chars_not_extracted=hf,
                content_control_chars_not_extracted=sdt, n_heading_paragraphs=heads, intro_before_first_heading=pre,
                n_sections_no_heading_fallback=int(heads == 0))

def policy_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--out", default="input_analysis")
    ap.add_argument("--results", default="results", help="baseline results folder (source of saved retrieval excerpts and archived context limits)")
    ap.add_argument("--tokenizers", action="store_true", help="tokenizer-only budget checks: 800/128 allowances, RAG k=1/3 from saved passages, embedding-input audit (downloads tokenizer files only)")
    ap.add_argument("--embeddings", action="store_true", help="ALSO run embedding inference to measure RAG k=5 (needs the embedding model weights; ~470 MB)")
    a = ap.parse_args(); repo = Path(a.repo); out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    from docx import Document
    apps = yaml.safe_load((repo / "configs/applications.yaml").read_text())["applications"]
    rows, srows, aud = [], [], []
    current = {app["id"]: (app, policy_sha256(repo / app["policy_path"])) for app in apps}
    for app in apps:
        doc = Document(str(repo / app["policy_path"])); secs = extract_like_pipeline(doc)
        if not secs:
            raise ValueError(f"No policy sections extracted from {app['policy_path']}")
        full = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        letters = len(AR.findall(full)) + len(LA.findall(full))
        rows.append(dict(app=app["name"], id=app["id"], chars=len(full), words=len(full.split()), n_sections=len(secs),
                         arabic_letter_share=round(len(AR.findall(full)) / max(letters, 1), 3), latin_letter_share=round(len(LA.findall(full)) / max(letters, 1), 3),
                         sec_chars_min=min(len(s["content"]) for s in secs), sec_chars_median=int(np.median([len(s["content"]) for s in secs])), sec_chars_max=max(len(s["content"]) for s in secs)))
        srows += [dict(app=app["name"], section=i, title=s["title"], chars=len(s["content"]), words=len(s["content"].split())) for i, s in enumerate(secs, 1)]
        aud.append(dict(app=app["name"], **audit_docx(doc, secs)))
    pol = pd.DataFrame(rows); pol.to_csv(out / "policy_summary.csv", index=False)
    pd.DataFrame(srows).to_csv(out / "section_lengths.csv", index=False); au = pd.DataFrame(aud); au.to_csv(out / "docx_extraction_audit.csv", index=False)
    full_rows, sec_rows, skipped = [], [], []
    results_path = Path(a.results)
    results_path = results_path if results_path.is_absolute() else repo / results_path
    for f in sorted(results_path.glob("raw/*/*/*/run_01.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        app_id = d["application"]["id"]
        if app_id not in current or d.get("policy", {}).get("sha256") != current[app_id][1]:
            skipped.append(dict(path=str(f), application=app_id, reason="Policy not configured or input hash changed"))
            continue
        m, s, lim = d["model"]["id"], d["strategy"], d["model"].get("context_limit_tokens")
        if d["unit"] == "full":
            t = d.get("token_lengths", {})
            if t.get("prompt_tokens"): full_rows.append(dict(model=m, strategy=s, app=d["application"]["name"], prompt_tokens=t["prompt_tokens"], limit=lim))
        else:
            for sp in d["section_predictions"]:
                if sp.get("prompt_tokens"): sec_rows.append(dict(model=m, strategy=s, app=d["application"]["name"], prompt_tokens=sp["prompt_tokens"], limit=lim))
    fr, sr = pd.DataFrame(full_rows, columns=["model", "strategy", "app", "prompt_tokens", "limit"]), pd.DataFrame(sec_rows, columns=["model", "strategy", "app", "prompt_tokens", "limit"]); fr.to_csv(out / "full_policy_prompt_tokens_saved.csv", index=False)
    pd.DataFrame(skipped, columns=["path", "application", "reason"]).to_csv(out / "excluded_saved_counts.csv", index=False)
    measured = set(fr.app)
    pd.DataFrame([dict(application=app["name"], reason="No matching saved full-policy token counts; run tokenizer analysis") for app in apps if app["name"] not in measured], columns=["application", "reason"]).to_csv(out / "missing_token_counts.csv", index=False)
    mat = []
    for allow in (800, 128):
        for (m, s), g in fr.groupby(["model", "strategy"]):
            mat.append(dict(unit="full-policy", model=m, strategy=s, allowance=allow, requests=len(g), fit=int((g.prompt_tokens + allow <= g.limit).sum()),
                            fits_prompt_alone=int((g.prompt_tokens <= g.limit).sum()), limit=int(g.limit.iloc[0]), max_prompt_tokens=int(g.prompt_tokens.max())))
        for (m, s), g in sr.groupby(["model", "strategy"]):
            mat.append(dict(unit="section", model=m, strategy=s, allowance=allow, requests=len(g), fit=int((g.prompt_tokens + allow <= g.limit).sum()),
                            fits_prompt_alone=int((g.prompt_tokens <= g.limit).sum()), limit=int(g.limit.iloc[0]), max_prompt_tokens=int(g.prompt_tokens.max())))
    mt = pd.DataFrame(mat, columns=["unit", "model", "strategy", "allowance", "requests", "fit", "fits_prompt_alone", "limit", "max_prompt_tokens"]); mt["fit_pct"] = (100 * mt.fit / mt.requests).round(1); mt.to_csv(out / "context_feasibility_matrix.csv", index=False)
    if not fr.empty:
        ov = fr.pivot_table(index=["model", "app"], columns="strategy", values="prompt_tokens")
        if {"full_rag", "full_norag"}.issubset(ov.columns):
            ov["rag_overhead"] = ov.full_rag - ov.full_norag
            ov.groupby("model").rag_overhead.agg(["min", "max"]).to_csv(out / "full_rag_overhead_tokens.csv")
    figures(out, pol, mt, fr)
    print(pol[["chars", "words", "n_sections"]].describe().round(0).loc[["min", "50%", "max"]]); print(au.drop(columns="app").sum(numeric_only=True))
    if a.tokenizers or a.embeddings:
        rf = None
        if a.embeddings:
            def rf(cfg):
                from pdpl.retrieval import build_retriever
                r = dict(cfg.retrieval); r["corpus_path"] = str(cfg.root / r["corpus_path"]); return build_retriever(r)
        requests, sm = measure_context_budget(repo, out / "tokenizer_checks", a.results, with_embeddings=a.embeddings, retriever_factory=rf)
        measured = requests[(requests.unit == "full") & requests.variant.isin(["norag", "rag_k3"])].copy()
        measured["strategy"] = measured.variant.map({"norag": "full_norag", "rag_k3": "full_rag"})
        measured["app"] = measured.app.map({app["id"]: app["name"] for app in apps})
        measured.to_csv(out / "full_policy_prompt_tokens_current.csv", index=False)
        figures(out, pol, mt, measured)
        print(sm[sm.allowance == 800].pivot_table(index=["model", "unit"], columns="variant", values="fit_pct").round(0))

def figures(out, pol, mt, fr):
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for name in ("context_feasibility_matrix", "context_budget_full", "context_budget_section"):
        for ext in ("png", "pdf"):
            old = figdir / f"{name}.{ext}"
            if old.exists(): old.unlink()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.axisbelow": True})
    height = max(5.5, .29 * len(pol) + 1.2)
    p = pol.sort_values("words", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, height), sharey=True)
    y = np.arange(len(p))
    ax[0].barh(y, p.words, color="#4477AA"); ax[0].set_xlabel("Number of words"); ax[0].set_title("Policy length")
    ax[1].barh(y, p.n_sections, color="#EE7733"); ax[1].set_xlabel("Number of sections"); ax[1].set_title("Policy sections")
    ax[0].set_yticks(y, p.app); ax[0].invert_yaxis(); ax[0].set_ylabel("Privacy policy")
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    for axis in ax:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True)); axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}")); axis.grid(axis="x", alpha=.2)
    fig.tight_layout()
    for ext in ("png",): fig.savefig(figdir / f"policy_length.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    al = fr[fr.model == "allam-7b"] if not fr.empty else fr
    if al.empty:
        for ext in ("png", "pdf"):
            old = figdir / f"allam_full_policy_tokens.{ext}"
            if old.exists(): old.unlink()
        print("ALLaM token plot not generated: no current matching measurements. Run --tokenizers --embeddings.")
        return
    wide = al.pivot_table(index="app", columns="strategy", values="prompt_tokens")
    sortcol = "full_norag" if "full_norag" in wide else wide.columns[0]
    wide = wide.sort_values(sortcol)
    limit_values = al.limit.dropna().unique()
    if len(limit_values) != 1: raise ValueError("ALLaM measurements must have one context limit")
    limit = int(limit_values[0])
    fig, ax = plt.subplots(figsize=(max(10, .48 * len(wide)), 6))
    x = np.arange(len(wide))
    for strategy, color, label in (("full_norag", "#4477AA", "Full policy without RAG"), ("full_rag", "#CC3311", "Full policy with RAG")):
        if strategy in wide: ax.plot(x, wide[strategy], marker="o", color=color, label=label, linewidth=1.7, markersize=4)
    ax.axhline(limit, color="#222222", linewidth=1.5, label=f"Context limit ({limit:,} tokens)")
    ax.axhline(limit - 128, color="#555555", linestyle=":", linewidth=1.5, label=f"Prompt budget: 128 response tokens ({limit - 128:,})")
    ax.axhline(limit - 800, color="#555555", linestyle="--", linewidth=1.5, label=f"Prompt budget: 800 response tokens ({limit - 800:,})")
    ax.set_xticks(x, wide.index, rotation=55, ha="right", fontsize=9)
    ax.set_xlabel("Privacy policy (ordered by prompt length without RAG)")
    ax.set_ylabel("Number of input tokens")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_title("ALLaM full-policy input length and context budgets")
    ax.grid(axis="y", alpha=.2); ax.legend(loc="upper left", fontsize=8.5)
    missing = len(pol) - len(wide)
    if missing or wide.isna().any().any():
        fig.text(.5, .01, f"Available current measurements: {len(wide)} of {len(pol)} policies. Missing points are not treated as zero.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04 if missing or wide.isna().any().any() else 0, 1, 1])
    for ext in ("png",): fig.savefig(figdir / f"allam_full_policy_tokens.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _auto_tokenizer(hf_id, revision=None, use_fast=False, trust_remote_code=False):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_id, revision=revision, use_fast=use_fast, trust_remote_code=trust_remote_code)

ALLOWANCES = (800, 128); RETRIEVAL_K = (1, 3, 5); WINDOWS = ((256, 32), (512, 64))

def measure_context_budget(repo, out, results="results", tokenizer_factory=_auto_tokenizer, with_embeddings=False, retriever_factory=None):
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from pdpl.config import load_config
    from pdpl.inputs import load_rubric, load_prompt, load_policy, render_full_prompt, render_section_prompt, parse_docx_to_texts_metadatas
    from pdpl.models import HFLocalModel
    repo, out = Path(repo), Path(out); (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(repo / "configs/experiment.yaml"), root=repo); rub = load_rubric(cfg.rubric_path)
    tpl = {n: load_prompt(n, s.prompt_path) for n, s in cfg.strategies.items()}
    sep_s, sep_f = cfg.retrieval["section_context_separator"], cfg.retrieval["full_context_separator"]
    saved = {}                                   
    for app in cfg.applications:
        for strat in ("section_rag", "full_rag"):
            f = Path(repo / results / "raw" / app.id / "llama-3.1-8b" / strat / "run_01.json")
            if f.exists():
                r = json.loads(f.read_text(encoding="utf-8"))
                if r.get("policy", {}).get("sha256") != policy_sha256(app.policy_path): continue
                saved[(app.id, strat)] = [(sp.get("retrieval") or {}).get("retrieved") for sp in r["section_predictions"]] if strat == "section_rag" else (r.get("retrieval") or {}).get("retrieved")
    retr = retriever_factory(cfg) if (with_embeddings and retriever_factory) else None
    policies = {a.id: load_policy(a.policy_path) for a in cfg.applications}
    rows = []
    for spec in [m for m in cfg.models if m.type == "hf_local"]:
        lm = HFLocalModel(spec); lm.tok = tokenizer_factory(spec.hf_id, spec.revision, spec.tokenizer_use_fast, spec.trust_remote_code)
        limit = int(spec.context_limit_tokens) if spec.context_limit_tokens else _archived_limit(repo, results, spec.id)   # limit recorded by the archived run
        for app in cfg.applications:
            pol = policies[app.id]
            hits_full = saved.get((app.id, "full_rag"))
            variants = {"norag": None}
            if hits_full:
                variants.update({f"rag_k{k}": hits_full[:k] for k in (1, 3) if len(hits_full) >= k})
            if retr:
                fresh = retr.retrieve(rub.text, 5)
                variants.update({f"rag_k{k}": fresh[:k] for k in (1, 3, 5) if len(fresh) >= k})
            for v, hits in variants.items():
                t = tpl["full_norag" if hits is None else "full_rag"]
                prompt = render_full_prompt(t, rub.text, pol.full_text, None if hits is None else sep_f.join(h["text"] for h in hits))
                rows.append(dict(model=spec.id, unit="full", variant=v, app=app.id, section=0, prompt_tokens=lm.count_tokens(prompt), limit=limit))
            sec_hits = saved.get((app.id, "section_rag"))
            for i, sec in enumerate(pol.sections):
                sv = {"norag": None}
                if sec_hits and i < len(sec_hits) and sec_hits[i]:
                    sv.update({f"rag_k{k}": sec_hits[i][:k] for k in (1, 3) if len(sec_hits[i]) >= k})
                if retr:
                    fresh = retr.retrieve(sec["content"], 5)
                    sv.update({f"rag_k{k}": fresh[:k] for k in (1, 3, 5) if len(fresh) >= k})
                for v, hits in sv.items():
                    t = tpl["section_norag" if hits is None else "section_rag"]
                    prompt = render_section_prompt(t, rub.text, sec["title"], sec["content"], None if hits is None else sep_s.join(h["text"] for h in hits))
                    rows.append(dict(model=spec.id, unit="section", variant=v, app=app.id, section=i + 1, prompt_tokens=lm.count_tokens(prompt), limit=limit))
    df = pd.DataFrame(rows); df.to_csv(out / "context_budget_requests.csv", index=False)
    summ = []
    for allow in ALLOWANCES:
        for (m, u, v), g in df.groupby(["model", "unit", "variant"]):
            summ.append(dict(model=m, unit=u, variant=v, allowance=allow, requests=len(g), fit=int((g.prompt_tokens + allow <= g.limit).sum()), fits_prompt_alone=int((g.prompt_tokens <= g.limit).sum()),
                             limit=int(g.limit.iloc[0]), max_prompt_tokens=int(g.prompt_tokens.max()), median_prompt_tokens=int(g.prompt_tokens.median())))
    sm = pd.DataFrame(summ); sm["fit_pct"] = (100 * sm.fit / sm.requests).round(1); sm.to_csv(out / "context_budget_summary.csv", index=False)
    et = tokenizer_factory(cfg.retrieval["embedding_model"], cfg.retrieval.get("embedding_revision")); maxlen = int(cfg.retrieval["embedding_max_length"]); n = lambda t: len(et(t, truncation=False)["input_ids"])
    emb = [dict(input="full-policy RAG query (rubric)", variant="current", n=1, max_tokens=n("query: " + rub.text), over_limit=int(n("query: " + rub.text) > maxlen))]
    secq = [n("query: " + sec["content"]) for p_ in policies.values() for sec in p_.sections]
    emb.append(dict(input="section RAG queries (all policies)", variant="current", n=len(secq), max_tokens=max(secq), over_limit=sum(x > maxlen for x in secq)))
    texts, _ = parse_docx_to_texts_metadatas(cfg.root / cfg.retrieval["corpus_path"]); ln = [n("passage: " + t) for t in texts]
    emb.append(dict(input="PDPL passages (heading sections)", variant="current heading chunks", n=len(ln), max_tokens=max(ln), over_limit=sum(x > maxlen for x in ln)))
    ids = et(" ".join(texts), truncation=False, add_special_tokens=False)["input_ids"]
    for w, o in WINDOWS:
        starts = list(range(0, max(len(ids) - o, 1), w - o)); emb.append(dict(input="PDPL passages (token windows)", variant=f"window {w}/overlap {o}", n=len(starts), max_tokens=min(w, len(ids)), over_limit=0))
    pd.DataFrame(emb).to_csv(out / "embedding_input_audit.csv", index=False)
    return df, sm

def _archived_limit(repo, results, model_id):
    for f in Path(repo / results / "raw").glob(f"*/{model_id}/*/run_01.json"):
        lim = json.loads(f.read_text(encoding="utf-8")).get("model", {}).get("context_limit_tokens")
        if lim: return int(lim)
    raise SystemExit(f"Cannot determine context limit for {model_id}: set context_limit_tokens in configs/models.yaml")

if __name__ == "__main__": main()
