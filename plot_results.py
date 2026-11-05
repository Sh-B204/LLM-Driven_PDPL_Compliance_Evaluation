import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from evaluate_results import confusion_counts, error_asymmetry, FIRST_RUN   # same exact-count and exact-test helpers as the evaluation export

W1, W2 = 3.5, 7.16                                   # IEEE column widths (in)
MODELS = ['allam-7b', 'llama-3.1-8b', 'qwen-2.5-7b', 'mistral-7b']
NAMES = {'allam-7b': 'ALLaM-7B', 'llama-3.1-8b': 'LLaMA-3.1-8B', 'qwen-2.5-7b': 'Qwen-2.5-7B', 'mistral-7b': 'Mistral-7B'}
MCOL = dict(zip(MODELS, ['#0072B2', '#D55E00', '#009E73', '#CC79A7']))
STRATS = ['section_norag', 'section_rag', 'full_norag', 'full_rag']
SLAB = {'section_norag': 'Section', 'section_rag': 'Section + RAG', 'full_norag': 'Full policy', 'full_rag': 'Full policy + RAG'}
SMARK = {'section_norag': ('o', True), 'section_rag': ('o', False), 'full_norag': ('s', True), 'full_rag': ('s', False)}
RULES = ['or', 'at_least_two', 'majority', 'and']
RLAB = {'or': 'OR (≥1 section)', 'at_least_two': '≥2 sections', 'majority': 'Majority', 'and': 'AND (all sections)'}
RMARK = {'or': 'o', 'at_least_two': '^', 'majority': 'D', 'and': 's'}
GREY, INK = '#8a8f98', '#222222'
OLD_FILES = ['table_available', 'table_matched', 'fig_rag_paired', 'fig_rag_effect', 'fig_application_accuracy', 'fig_score_calibration',
             'fig_criterion_agreement', 'fig_criterion_errors', 'fig_aggregation', 'fig_stability']


def norm(x): return re.sub('[^a-z0-9]', '', str(x).lower())


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'savefig.facecolor': 'white'})


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=450, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)
    print('  wrote', name + '.png')


def panel(ax, tag):
    ax.text(0, 1.04, f'({tag})', transform=ax.transAxes, fontweight='bold', fontsize=7.5)

def mark(ax, x, y, model, strat, s=26, z=4, **kw):
    m, filled = SMARK[strat]; c = MCOL[model]
    return ax.scatter(x, y, s=s, marker=m, facecolor=c if filled else 'white', edgecolor=c, linewidths=1.0, zorder=z, **kw)


def cfg_rows(sub=None):
    rows, y = [], 0
    for m in MODELS:
        rows.append((y, m, None)); y += 1
        for s in STRATS: rows.append((y, m, s)); y += 1
    return rows


def ylabels(ax, rows, nmap=None):
    ys = [r[0] for r in rows]
    ax.set_yticks(ys)
    labs = []
    for _, m, s in rows:
        if s is None: labs.append(NAMES[m])
        else:
            n = nmap.get((m, s)) if nmap else None
            labs.append(SLAB[s] + (f' (n={n})' if n is not None and n < 21 else ''))
    ax.set_yticklabels(labs)
    for t, (_, m, s) in zip(ax.get_yticklabels(), rows):
        if s is None: t.set_fontweight('bold'); t.set_color(MCOL[m])
    ax.set_ylim(len(rows) - .4, -.8); ax.tick_params(axis='y', length=0)


def only_main(df, what):
    if 'model' not in df: return df
    bad = sorted(set(df.model) - set(MODELS))
    if bad: print(f'  NOTE: {what}: excluding models outside the four-model study: {bad}')
    return df[df.model.isin(MODELS)].copy()


def load_expert(repo, codes):
    p = Path(repo) / 'data/ground_truth/PDPL_Results.xlsx'
    if not p.exists(): return None, p
    gt = pd.read_excel(p, sheet_name='Expert Analysis')
    cols = {re.match(r'\s*([A-D]\d+)\.', str(c)).group(1): c for c in gt.columns[1:] if re.match(r'\s*([A-D]\d+)\.', str(c))}
    gt = gt[gt.iloc[:, 0].notna()]
    return {norm(r.iloc[0]): np.array([r[cols[c]] for c in codes], float) for _, r in gt.iterrows()}, p


def load_all(e, repo):
    rd = lambda n: pd.read_csv(e / n)
    D = {n: only_main(rd(n + '.csv'), n) for n in ['grand_table', 'coverage', 'criterion_level', 'application_percentages', 'aggregation_rules',
                                                   'repetition_agreement', 'paired_rag_effect', 'policy_cluster_ci', 'final_vectors_15_items', 'failed_records']}
    D['constant_baselines'] = rd('constant_baselines.csv'); D['section_vectors'] = only_main(rd('section_vectors.csv'), 'section_vectors')
    D['provenance_summary'] = rd('provenance_summary.csv')
    for n in ['pooled_confusion_matrices', 'error_asymmetry_run1']:                 # written by evaluate_results.py; used only for cross-checks
        D[n] = only_main(rd(n + '.csv'), n) if (e / (n + '.csv')).exists() else None
    try: D['provenance_exclusions'] = rd('provenance_exclusions.csv')
    except pd.errors.EmptyDataError: D['provenance_exclusions'] = pd.DataFrame()
    fv = D['final_vectors_15_items']; D['codes'] = [c for c in fv if re.fullmatch('[A-D][0-9]+', c)]
    D['n_runs'] = int(D['grand_table'].n_runs.max())
    # primary rule only (OR for section strategies; 'full' for full-policy), and only policies with ALL repetitions (same rule as evaluate_results)
    f = fv[fv.rule.isin(['or', 'full'])].copy()
    cnt = f.groupby(['model', 'strategy', 'app']).run.transform('nunique')
    D['vec'] = f[cnt == D['n_runs']].copy()
    D['all_apps'] = sorted(fv.app.unique())
    D['expert'], D['gt_path'] = load_expert(repo, D['codes'])
    return D


def config_cells(D, run=None):
    codes, ex, out = D['codes'], D['expert'], {}
    for (m, s), d in D['vec'].groupby(['model', 'strategy']):
        if run is not None: d = d[d.run == run]
        if d.empty: continue
        out[m, s] = (np.stack([ex[norm(a)] for a in d.app]).astype(int), d[codes].to_numpy(int), d.app.nunique(), d.run.nunique())
    return out


def asymmetry_table(D):
    rows = []
    for (m, s), (Y, P, n, _) in config_cells(D, run=FIRST_RUN).items():
        c = confusion_counts(Y, P); rows.append(dict(model=m, strategy=s, n_policies=n, n_decisions=int(Y.size), **error_asymmetry(c['fp'], c['fn'])))
    if not rows: raise SystemExit(f'Repetition {FIRST_RUN} not found; the error-asymmetry test needs it')
    return pd.DataFrame(rows)


def validate(D):
    E, Wn, M = [], [], []
    prov = D['provenance_summary']; ok = {'archived_protocol_1.1.0', 'protocol_2.x_verified'}
    bad = [l for l in prov.provenance_label if l not in ok]
    if bad: Wn.append(f'provenance labels outside scored set present in summary: {bad} (they should be in exclusions only)')
    if len(D['provenance_exclusions']): Wn.append(f'{len(D["provenance_exclusions"])} records were excluded by provenance validation (never scored, never zero)')
    cov, vec, fl = D['coverage'], D['vec'], D['failed_records']
    for _, r in cov.iterrows():
        v = vec[(vec.model == r.model) & (vec.strategy == r.strategy)]
        if len(v) != r.successful_records: E.append(f'coverage mismatch {r.model}/{r.strategy}: vectors {len(v)} vs coverage.successful_records {r.successful_records}')
        if v.app.nunique() != r.policies_complete_5_runs: E.append(f'complete-policy mismatch {r.model}/{r.strategy}: {v.app.nunique()} vs {r.policies_complete_5_runs}')
    v = vec.copy(); v['pp'] = v[D['codes']].mean(axis=1) * 100
    pp = v.groupby(['app', 'model', 'strategy']).pp.mean().rename('chk').reset_index().merge(D['application_percentages'], on=['app', 'model', 'strategy'])
    if len(pp) != len(D['application_percentages']): E.append('application_percentages rows not all reproducible from saved vectors')
    elif (pp.chk - pp.pred_pct_mean).abs().max() > 1e-6: E.append('predicted % from saved vectors differs from application_percentages.csv')
    if D['expert'] is None:
        M.append(f'MISSING INPUT: {D["gt_path"]} (sheet "Expert Analysis"). Needed for F2 (per-application accuracy), the per-policy dots in F4 and for the '
                 'accuracy re-computation check. Nothing is estimated in its place; these panels are skipped.')
    else:
        ex = D['expert']; ap = D['application_percentages'].drop_duplicates('app')
        for _, r in ap.iterrows():
            if abs(ex[norm(r.app)].mean() * 100 - r.expert_pct) > 1e-6: E.append(f'expert % mismatch for {r.app}')
        g = D['grand_table']; g = g[g.subset == 'available']
        for (m, s), d in vec.groupby(['model', 'strategy']):
            accs = []
            for r_, dr in d.groupby('run'):
                accs.append(np.mean(np.concatenate([(dr[dr.app == a][D['codes']].to_numpy(float)[0] == ex[norm(a)]) for a in dr.app])))
            ref = g[(g.model == m) & (g.strategy == s)].accuracy_mean.iloc[0]
            if abs(np.mean(accs) - ref) > 1e-9: E.append(f'accuracy re-computation differs from grand_table for {m}/{s}: {np.mean(accs):.6f} vs {ref:.6f}')
        # RAG paired means reproduce paired_rag_effect.csv
        acc = per_policy_accuracy(D)
        for _, r in D['paired_rag_effect'].iterrows():
            a, b = acc[(acc.model == r.model) & (acc.strategy == f'{r.unit}_rag')].set_index('app').acc, acc[(acc.model == r.model) & (acc.strategy == f'{r.unit}_norag')].set_index('app').acc
            ks = a.index.intersection(b.index)
            if len(ks) != r.n_policies or abs((a[ks] - b[ks]).mean() - r.mean_diff_accuracy) > 1e-9: E.append(f'paired RAG recomputation differs for {r.model}/{r.unit}')
        for name, run in [('pooled_confusion_matrices', None), ('error_asymmetry_run1', FIRST_RUN)]:
            ref = D[name]
            if ref is None: Wn.append(f'{name}.csv not found in the evaluation folder (re-run evaluate_results.py); values are recomputed here from saved vectors but not cross-checked'); continue
            ref = ref.set_index(['model', 'strategy'])
            for (m, s), (Y, P, n, _) in config_cells(D, run=run).items():
                if (m, s) not in ref.index: E.append(f'{name}.csv has no row for {m}/{s}'); continue
                c, r = confusion_counts(Y, P), ref.loc[(m, s)]
                keys = ['tn', 'fp', 'fn', 'tp'] if run is None else ['fp', 'fn']
                if any(int(r[k]) != c[k] for k in keys) or int(r.n_policies) != n: E.append(f'{name}.csv differs from counts recomputed from saved vectors for {m}/{s}')
        crit_ref = D['criterion_level'].set_index(['model', 'strategy', 'criterion'])
        for (m, s), (Y, P, n, _) in config_cells(D).items():
            for j, c in enumerate(D['codes']):
                r = crit_ref.loc[(m, s, c)]
                if abs((Y[:, j] == P[:, j]).mean() - r.agreement) > 1e-9 or abs(Y[:, j].mean() - r.expert_pos) > 1e-9: E.append(f'criterion agreement/prevalence differs from criterion_level.csv for {m}/{s}/{c}')
    cl = D['criterion_level']
    und = cl[(cl.expert_pos == 0) & (cl.fn_rate == 0)]
    if len(und): Wn.append(f'criterion_level.csv stores FNR=0 where the expert has no positives ({sorted(und.criterion.unique())}): treated as UNDEFINED (NaN) here; same for FPR when expert_pos==1')
    if len(D['constant_baselines']) and not D['constant_baselines'].subset.str.startswith(('available', 'matched')).all(): Wn.append('unexpected constant_baselines subset labels')
    full = D['grand_table'][(D['grand_table'].subset == 'available') & (D['grand_table'].n_policies == len(D['all_apps']))]
    ep = cl[cl.set_index(['model', 'strategy']).index.isin(full.set_index(['model', 'strategy']).index)].groupby('criterion').expert_pos.nunique()
    if (ep > 1).any(): E.append('expert prevalence differs between 21-policy configurations')
    return E, Wn, M


def per_policy_accuracy(D):
    ex, codes = D['expert'], D['codes']; v = D['vec']
    acc = np.array([np.mean(r[codes].to_numpy(float) == ex[norm(r.app)]) for _, r in v.iterrows()])
    return v.assign(acc=acc).groupby(['model', 'strategy', 'app']).acc.mean().reset_index()


def strategy_handles():
    return [Line2D([],[],marker=SMARK[s][0],ls='',mfc=GREY if SMARK[s][1] else 'white',mec=GREY,ms=5,label=SLAB[s]) for s in STRATS]

def final_overall(D,out):
    g=D['grand_table'];g=g[g.subset=='available'].set_index(['model','strategy'])
    ci=D['policy_cluster_ci'].set_index(['model','strategy']);acc=per_policy_accuracy(D)
    rows=cfg_rows();fig,ax=plt.subplots(figsize=(4.9,6.1));rng=np.random.default_rng(2026);nmap={}
    for y,m,s in rows:
        if s is None:continue
        r=g.loc[(m,s)];nmap[m,s]=int(r.n_policies);d=acc[(acc.model==m)&(acc.strategy==s)]
        ax.scatter(d.acc*100,y+rng.uniform(-.16,.16,len(d)),s=8,c=MCOL[m],alpha=.23,edgecolors='none',zorder=1)
        c=ci.loc[(m,s)];ax.plot([100*c.acc_lo,100*c.acc_hi],[y,y],c=MCOL[m],lw=1.8,zorder=3)
        mark(ax,100*r.accuracy_mean,y,m,s,s=26,z=4)
    ylabels(ax,rows,nmap);ax.set_xlim(-2,102);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('Criterion accuracy (%)');ax.grid(axis='x',alpha=.15,lw=.5)
    h=strategy_handles()+[Line2D([],[],marker='o',ls='',mfc=GREY,mec='none',alpha=.35,ms=3,label='One application (mean of 5 runs)'),Line2D([],[],color=GREY,lw=1.8,label='Policy-cluster 95% interval')]
    fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.53,.015),ncol=2,frameon=False,fontsize=6.5)
    save(fig,out,'F1_overall_accuracy')

def final_applications(D,out):
    a=per_policy_accuracy(D);apps=sorted(D['all_apps']);fig,axs=plt.subplots(1,4,figsize=(W2,6.8),sharey=True)
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#e5e5e5')
    allcols=[]
    for ax,m in zip(axs,MODELS):
        t=a[a.model==m].pivot(index='app',columns='strategy',values='acc').reindex(index=apps,columns=STRATS)*100;allcols.append(t)
        im=ax.imshow(t,aspect='auto',vmin=0,vmax=100,cmap=cmap)
        ax.set_xticks(range(4),['S','S+R','F','F+R'],rotation=45,ha='right');ax.set_yticks(range(len(apps)),apps);ax.set_title(NAMES[m],color=MCOL[m],fontsize=7)
        for i in range(len(t)):
            for j in range(4):
                v=t.iloc[i,j];ax.text(j,i,'—' if pd.isna(v) else f'{v:.0f}',ha='center',va='center',fontsize=4.8,color='white' if v>70 else INK)
    fig.subplots_adjust(wspace=.10,right=.91,bottom=.1);ca=fig.add_axes([.93,.2,.012,.6]);fig.colorbar(im,cax=ca,label='Mean accuracy (%)')
    fig.text(.54,.025,'S: section · F: full policy · +R: RAG · grey: unavailable',ha='center',fontsize=6)
    save(fig,out,'F2_application_accuracy')
    a.assign(accuracy_pct=a.acc*100).to_csv(out/'application_accuracy_values.csv',index=False)

def final_rubrics(D,out, binary=True):
    from matplotlib.colors import LinearSegmentedColormap
    codes=D['codes'];v=D['vec'];ex=D['expert'];rows=[];counts=[]
    configs=[(m,s) for m in MODELS for s in STRATS]
    for m,s in configs:
        d=v[(v.model==m)&(v.strategy==s)];Y=np.stack([ex[norm(a)] for a in d.app]);P=d[codes].to_numpy(float)
        rows.append((Y==P).mean(0));counts.append(d.app.nunique())
    A=np.array(rows);cmap=LinearSegmentedColormap.from_list('expert_agreement',['#c94b4b','#fff6eb','#39865a'])
    fig,ax=plt.subplots(figsize=(W2,5.1));im=ax.imshow(A,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set_xticks(range(len(codes)),codes);ax.set_yticks(range(16),[f'{NAMES[m]} · {SLAB[s]} (n={n})' for (m,s),n in zip(configs,counts)],fontsize=6)
    for i in range(16):
        for j in range(len(codes)):ax.text(j,i,f'{A[i,j]*100:.0f}',ha='center',va='center',fontsize=5,color='white' if A[i,j]<.15 or A[i,j]>.85 else INK)
    for y in [3.5,7.5,11.5]:ax.axhline(y,c='white',lw=2)
    ax.set_xticks(np.arange(-.5,len(codes),1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True);ax.grid(which='minor',color='white',lw=.4);ax.tick_params(which='minor',length=0)
    cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.025,ticks=[0,.5,1],label='Agreement with expert (%)');cb.ax.set_yticklabels(['0','50','100']);ax.set_xlabel('Rubric criterion');save(fig,out,'F3_rubric_agreement')
    pd.DataFrame(A,index=[f'{m}/{s}' for m,s in configs],columns=codes).to_csv(out/'rubric_agreement_values.csv')
    if not binary: return
    folder=out/'application_binary_grids';folder.mkdir(exist_ok=True);exports=[]
    if True:
        for app in sorted(D['all_apps']):
            gt=ex[norm(app)];B=np.full((17,len(codes)),np.nan);txt=[['']*len(codes) for _ in range(17)];txt[0]=[str(int(x)) for x in gt];labs=['Expert labels']
            for i,(m,s) in enumerate(configs,1):
                labs.append(f'{NAMES[m]} · {SLAB[s]}');d=v[(v.app==app)&(v.model==m)&(v.strategy==s)]
                if d.empty:continue
                P=d[codes].to_numpy(int);B[i]=(P==gt).mean(0);ones=P.sum(0);major=(ones>len(P)/2).astype(int)
                for j,c in enumerate(codes):
                    mixed=0<ones[j]<len(P);tie=2*ones[j]==len(P);txt[i][j]=('tie' if tie else str(major[j]))+('*' if mixed else '')
                    exports.append(dict(app=app,model=m,strategy=s,criterion=c,expert=int(gt[j]),positive_runs=int(ones[j]),runs=len(P),agreement=B[i,j],majority=None if tie else int(major[j])))
            cm=cmap.copy();cm.set_bad('#e1e1e1');fig,ax=plt.subplots(figsize=(W2,4.9));ax.imshow(np.ma.masked_invalid(B),vmin=0,vmax=1,cmap=cm,aspect='auto')
            for i in range(17):
                for j in range(len(codes)):
                    t=txt[i][j] if i==0 or pd.notna(B[i,j]) else '—';ax.text(j,i,t,ha='center',va='center',fontsize=5.5,color=INK if i==0 or pd.isna(B[i,j]) or .2<B[i,j]<.8 else 'white')
            ax.set_xticks(range(len(codes)),codes);ax.set_yticks(range(17),labs,fontsize=5.8);ax.set_title(app,fontsize=9);ax.axhline(.5,c='white',lw=2)
            for y in [4.5,8.5,12.5]:ax.axhline(y,c='white',lw=1.7)
            ax.set_xticks(np.arange(-.5,len(codes),1),minor=True);ax.set_yticks(np.arange(-.5,17,1),minor=True);ax.grid(which='minor',color='white',lw=.4);ax.tick_params(which='minor',length=0)
            fig.text(.5,.005,'Digit: majority prediction · *: changed across runs · green: agreement · red: disagreement\nShade: proportion of runs agreeing with expert · grey: expert row / unavailable',ha='center',fontsize=6)
            save(fig,folder,norm(app))
    pd.DataFrame(exports).to_csv(folder/'binary_grid_values.csv',index=False)


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


def select_rule(D, rule, boot):
    f=D['final_vectors_15_items'];v=f[(f.rule==rule)|(f.rule=='full')].copy()
    v=v[v.groupby(['model','strategy','app']).run.transform('nunique')==D['n_runs']].copy()
    D['vec']=v
    codes=D['codes'];ex=D['expert'];v['key']=v.app.map(norm);v['vec']=[r.to_numpy(int) for _,r in v[codes].iterrows()]
    cfg=['model','strategy'];common=set.intersection(*[set(g.key) for _,g in v.groupby(cfg)])
    D['grand_table']=pd.concat([summarize(v,ex,cfg,'available'),summarize(v[v.key.isin(common)],ex,cfg,'matched')],ignore_index=True)
    rng=np.random.default_rng(2026);cis=[];pr=[]
    for (m,s),d in v.groupby(cfg):
        apps=sorted(d.app.unique());runs=sorted(d.run.unique());Y=np.stack([ex[norm(a)] for a in apps]);P=np.stack([d[d.run==r].set_index('app').loc[apps,codes].to_numpy(int) for r in runs]);bs=[]
        for _ in range(boot):
            ix=rng.integers(0,len(apps),len(apps));bs.append(np.mean(P[:,ix,:]==Y[None,ix,:]))
        lo,hi=np.percentile(bs,[2.5,97.5]);cis.append(dict(model=m,strategy=s,n_policies=len(apps),acc_lo=lo,acc_hi=hi))
    D['policy_cluster_ci']=pd.DataFrame(cis)
    a=per_policy_accuracy(D)
    for m in MODELS:
        for unit in ['section','full']:
            q=a[a.model==m].pivot(index='app',columns='strategy',values='acc').reindex(columns=[unit+'_norag',unit+'_rag']).dropna();diff=(q.iloc[:,1]-q.iloc[:,0]).to_numpy();bs=[diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(boot)];lo,hi=np.percentile(bs,[2.5,97.5]);pr.append(dict(model=m,unit=unit,n_policies=len(diff),mean_diff_accuracy=diff.mean(),ci_lo=lo,ci_hi=hi))
    D['paired_rag_effect']=pd.DataFrame(pr)
    return D

def rag_bars(D,out):
    a=per_policy_accuracy(D);pr=D['paired_rag_effect'];rows=[]
    for m in MODELS:
        for unit in ['section','full']:
            p=a[a.model==m].pivot(index='app',columns='strategy',values='acc').reindex(columns=[unit+'_norag',unit+'_rag']).dropna();r=pr[(pr.model==m)&(pr.unit==unit)].iloc[0]
            diff=(p.iloc[:,1]-p.iloc[:,0]).mean()*100
            if len(p)!=r.n_policies or not np.isclose(diff,r.mean_diff_accuracy*100,atol=1e-8):raise ValueError('Paired RAG table mismatch')
            rows.append(dict(model=m,unit=unit,n_policies=len(p),no_rag_pct=p.iloc[:,0].mean()*100,rag_pct=p.iloc[:,1].mean()*100,change_pp=diff,ci_lo_pp=r.ci_lo*100,ci_hi_pp=r.ci_hi*100))
    t=pd.DataFrame(rows);t.to_csv(out/'RAG_paired_values.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(W2,3.1),sharey=True)
    lim=max(abs(t.ci_lo_pp.min()),abs(t.ci_hi_pp.max()));lim=np.ceil((lim+2)/5)*5
    for ax,unit,title in zip(axs,['section','full'],['Section strategies','Full-policy strategies']):
        d=t[t.unit==unit].set_index('model').loc[MODELS];x=np.arange(4)
        for i,(m,r) in enumerate(d.iterrows()):
            ax.bar(i,r.change_pp,color=MCOL[m],width=.55,zorder=3);ax.plot([i,i],[r.ci_lo_pp,r.ci_hi_pp],c=INK,lw=.8,zorder=4)
            y=r.ci_hi_pp+.7 if r.change_pp>=0 else r.ci_lo_pp-.7
            ax.text(i,y,f'{r.change_pp:+.2f}',ha='center',va='bottom' if r.change_pp>=0 else 'top',fontsize=6.5)
        ax.axhline(0,c=INK,lw=.7);ax.set_ylim(-lim,lim);ax.set_xticks(x,[NAMES[m]+f'\n(n={int(d.loc[m,"n_policies"])})' for m in MODELS],rotation=25,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.15,zorder=0)
    axs[0].set_ylabel('Mean accuracy change (percentage points)');fig.tight_layout();save(fig,out,'F4a_RAG_change_bars')
    fig,axs=plt.subplots(1,2,figsize=(W2,3.1),sharey=True)
    for ax,unit,title in zip(axs,['section','full'],['Section strategies','Full-policy strategies']):
        d=t[t.unit==unit].set_index('model').loc[MODELS];x=np.arange(4)
        for i,(m,r) in enumerate(d.iterrows()):
            ax.bar(i-.17,r.no_rag_pct,width=.32,color=MCOL[m]);ax.bar(i+.17,r.rag_pct,width=.32,facecolor='white',edgecolor=MCOL[m],hatch='///',lw=.9)
            for xx,val in [(i-.17,r.no_rag_pct),(i+.17,r.rag_pct)]:ax.text(xx,val+1,f'{val:.1f}',ha='center',fontsize=5.5)
        ax.set_xticks(x,[NAMES[m]+f'\n(n={int(d.loc[m,"n_policies"])})' for m in MODELS],rotation=25,ha='right');ax.set_ylim(0,100);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[0].set_ylabel('Mean criterion accuracy (%)');fig.legend(handles=[Patch(facecolor=GREY,label='Without RAG'),Patch(facecolor='white',edgecolor=GREY,hatch='///',label='With RAG')],loc='upper center',bbox_to_anchor=(.5,1.07),ncol=2,frameon=False);fig.tight_layout();save(fig,out,'F4b_RAG_side_by_side_bars')

def aggregation_figure(D,out):
    d=D['aggregation_rules'];fig,axs=plt.subplots(1,2,figsize=(W2,3.2),sharey=True)
    for ax,s,title in zip(axs,['section_norag','section_rag'],['Section: without RAG','Section: with RAG']):
        for m in MODELS:
            q=d[(d.model==m)&(d.strategy==s)].set_index('rule').loc[RULES];ax.plot(range(4),q.balanced_accuracy_mean*100,color=MCOL[m],marker='o',lw=1.2,ms=4,label=NAMES[m])
        ax.axhline(50,c=GREY,ls=':',lw=.8);ax.set_xticks(range(4),['OR','≥2 sections','Majority','AND']);ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('Balanced accuracy (%)');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1.06),ncol=4,frameon=False);fig.tight_layout();save(fig,out,'F5_aggregation_rules')
    d[['model','strategy','rule','accuracy_mean','accuracy_sd','balanced_accuracy_mean','balanced_accuracy_sd','precision_mean','recall_mean','mcc_mean']].to_csv(out/'aggregation_comparison.csv',index=False)

def stability_figure(D,out):
    rows=cfg_rows();fig,ax=plt.subplots(figsize=(4.9,5.5));export=[]
    for y,m,s in rows:
        if s is None:continue
        d=D['vec'][(D['vec'].model==m)&(D['vec'].strategy==s)];vals=[]
        for app,q in d.groupby('app'):
            P=q.sort_values('run')[D['codes']].to_numpy(int)
            vals += [(P[i]==P[j]).mean() for i in range(len(P)) for j in range(i+1,len(P))]
        v=np.mean(vals)*100 if vals else np.nan
        mark(ax,v,y,m,s,s=25);export.append(dict(model=m,strategy=s,agreement_pct=v,n_policies=d.app.nunique()))
    ylabels(ax,rows);ax.set_xlim(90,100.5);ax.set_xticks([90,92,94,96,98,100]);ax.set_xlabel('Pairwise agreement between repetitions (%)');ax.grid(axis='x',alpha=.15)
    fig.legend(handles=strategy_handles(),loc='upper center',bbox_to_anchor=(.53,.01),ncol=2,frameon=False);save(fig,out,'F6_repetition_consistency');pd.DataFrame(export).to_csv(out/'repetition_consistency_values.csv',index=False)

def write_main_table(D,out):
    g=D['grand_table']
    metrics=[('accuracy',100,2,r'Accuracy (\%)'),('balanced_accuracy',100,2,r'BA (\%)'),('f1_class_macro',100,2,r'Macro-F1 (\%)'),('mcc',1,3,'MCC'),('kappa',1,3,r'$\kappa$'),('precision',100,2,r'Precision (\%)'),('recall',100,2,r'Recall (\%)'),('specificity',100,2,r'Specificity (\%)')]
    def tex(t):return str(t).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
    def cell(r,t,key,scale,dec):
        v=r[key+'_mean'];sd=r[key+'_sd']
        if pd.isna(v):return '--'
        value=f'{v*scale:.{dec}f} $\\pm$ '+('--' if pd.isna(sd) else f'{sd*scale:.{dec}f}')
        return r'\textbf{'+value+'}' if np.isclose(v,t[key+'_mean'].max(),rtol=0,atol=1e-12) else value
    for subset in ['available','matched']:
        if g[g.subset==subset].empty:continue
        t=g[g.subset==subset].set_index(['model','strategy']).reindex(pd.MultiIndex.from_product([MODELS,STRATS],names=['model','strategy'])).dropna(subset=['n_policies']).reset_index()
        cap=('Model performance across four strategies' if subset=='available' else 'Model performance on the common policy subset')+r', reported as mean $\pm$ sample SD over '+str(int(t.n_runs.max()))+r' runs. $n$ denotes evaluated policies; section predictions use '+tex(D.get('aggregation','or').upper())+r'. Bold indicates column maxima.'
        L=[r'\begin{table*}[!t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{2pt}',r'\caption{'+cap+'}',r'\label{tab:llm_'+subset+'}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llrcccccccc}',r'\toprule',r'Model & Strategy & $n$ & '+' & '.join(h for _,_,_,h in metrics)+r' \\',r'\midrule'];records=[]
        for i,r in t.iterrows():
            if i and t.iloc[i-1].model!=r.model:L.append(r'\midrule')
            lead=tex(NAMES[r.model] if i==0 or t.iloc[i-1].model!=r.model else '')+' & '+tex(SLAB[r.strategy])+' & '+str(int(r.n_policies))+' & '
            L.append(lead+' & '.join(cell(r,t,k,sc,d) for k,sc,d,_ in metrics)+r' \\')
            rec=dict(model=NAMES[r.model],strategy=SLAB[r.strategy],n_policies=int(r.n_policies))
            for k,_,_,_ in metrics:rec.update({k+'_mean':r[k+'_mean'],k+'_sd':r[k+'_sd']})
            records.append(rec)
        L += [r'\bottomrule',r'\end{tabular}',r'}',r'\end{table*}']
        (out/f'T_{subset}.tex').write_text('\n'.join(L),encoding='utf-8');pd.DataFrame(records).to_csv(out/f'T_{subset}.csv',index=False)
    q=asymmetry_table(D).set_index(['model','strategy']).reindex(pd.MultiIndex.from_product([MODELS,STRATS],names=['model','strategy'])).dropna(subset=['n_policies']).reset_index()
    cap=r'Error asymmetry in the first run. FP and FN are counts on successfully evaluated policies ($n$). $p$ is the two-sided exact McNemar test of equal FP and FN frequencies. Tests are exploratory, unadjusted for multiple comparisons and within-policy dependence.'
    L=[r'\begin{table}[!t]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\caption{'+cap+'}',r'\label{tab:error_asymmetry}',r'\begin{tabular}{llrrrc}',r'\toprule',r'Model & Strategy & $n$ & FP & FN & $p$ \\',r'\midrule']
    for i,r in q.iterrows():
        if i and q.iloc[i-1].model!=r.model:L.append(r'\midrule')
        pv='--' if pd.isna(r.p_value) else (r'$<0.001$' if r.p_value<.001 else f'{r.p_value:.3f}')
        L.append(tex(NAMES[r.model] if i==0 or q.iloc[i-1].model!=r.model else '')+' & '+tex(SLAB[r.strategy])+' & '+str(int(r.n_policies))+' & '+str(int(r.fp))+' & '+str(int(r.fn))+' & '+pv+r' \\')
    L += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'T_asymmetry.tex').write_text('\n'.join(L),encoding='utf-8');q.to_csv(out/'T_asymmetry.csv',index=False)

def criterion_table(D):
    rows=[]
    for (m,s),(Y,P,n,_) in config_cells(D).items():
        for j,c in enumerate(D['codes']):
            y,q=Y[:,j],P[:,j];k=confusion_counts(y,q);neg,pos=k['fp']+k['tn'],k['fn']+k['tp']
            rows.append(dict(model=m,strategy=s,criterion=c,n_policies=n,agreement=(y==q).mean(),prevalence_baseline=y.mean(),
                             fpr=k['fp']/neg if neg else np.nan,fnr=k['fn']/pos if pos else np.nan))
    return pd.DataFrame(rows)

def write_criterion_table(D,out):
    per=criterion_table(D)
    per=per[per.n_policies==len(D['all_apps'])].copy()
    if per.empty: raise SystemExit('No full-coverage configuration is available for Table 6')
    ncfg=per.groupby(['model','strategy']).ngroups
    npol=len(D['all_apps'])
    t=per.groupby('criterion').agg(agreement=('agreement','mean'),prevalence_baseline=('prevalence_baseline','mean'),fpr=('fpr','mean'),n_configs_fpr=('fpr','count'),
                                   fnr=('fnr','mean'),n_configs_fnr=('fnr','count'),n_configs=('agreement','size')).reindex(D['codes']).reset_index()
    positive_counts=np.rint(t.prevalence_baseline*npol).astype(int)
    csv=pd.DataFrame(dict(criterion=t.criterion,expert_positive_policies=positive_counts,n_policies=npol,agreement_pct=t.agreement*100,prevalence_baseline_pct=t.prevalence_baseline*100,
                          agreement_minus_baseline_pp=(t.agreement-t.prevalence_baseline)*100,fpr_pct=t.fpr*100,fnr_pct=t.fnr*100,
                          n_configs=t.n_configs,n_configs_fpr=t.n_configs_fpr,n_configs_fnr=t.n_configs_fnr));csv.to_csv(out/'T_criterion_errors.csv',index=False)
    def f(v,n):
        if pd.isna(v):return '--'                                          # rate undefined in every configuration
        return f'{v*100:.2f}'+(r'$^{\dagger}$' if n<ncfg else '')
    cap=(r'Criterion agreement and error rates (\%): means over '+str(ncfg)+r' configurations covering all '+str(npol)+r' policies, pooling repetitions within each configuration. Positives are expert-positive policy counts. The prevalence baseline is the accuracy of always predicting disclosed. Undefined rates remain missing (--); $^{\dagger}$ indicates a mean over fewer configurations. Section strategies use '+D.get('aggregation','or').upper().replace('_',r'\_')+r'.')
    L=[r'\begin{table}[t]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\caption{'+cap+'}',r'\label{tab:criterion_errors}',r'\begin{tabular}{lccccc}',r'\toprule',
       r'Item & Positives & \shortstack{Agreement\\(\%)} & \shortstack{Baseline\\(\%)} & FPR & FNR \\',r'\midrule']
    for i,r in t.iterrows():L.append(f'{r.criterion} & {positive_counts.iloc[i]}/{npol} & {r.agreement*100:.2f} & {r.prevalence_baseline*100:.2f} & {f(r.fpr,r.n_configs_fpr)} & {f(r.fnr,r.n_configs_fnr)} \\\\')
    (out/'T_criterion_errors.tex').write_text('\n'.join(L+[r'\bottomrule',r'\end{tabular}',r'\end{table}']),encoding='utf-8')


def write_rag_errors(D, out):
    v, codes, expert = D['vec'], D['codes'], D['expert']
    rows = []
    for model in MODELS:
        for unit in ('section', 'full'):
            no = v[(v.model == model) & (v.strategy == unit + '_norag')]
            rag = v[(v.model == model) & (v.strategy == unit + '_rag')]
            keys = sorted(set(no.app) & set(rag.app))
            row = dict(model=model, unit=unit, n_policies=len(keys))
            for name, data in [('without_rag', no), ('with_rag', rag)]:
                data = data[data.app.isin(keys)]
                y = np.stack([expert[norm(app)] for app in data.app]).ravel()
                pred = data[codes].to_numpy(int).ravel()
                c = confusion_counts(y, pred)
                row['fpr_' + name] = 100*c['fp']/(c['fp']+c['tn']) if c['fp']+c['tn'] else np.nan
                row['fnr_' + name] = 100*c['fn']/(c['fn']+c['tp']) if c['fn']+c['tp'] else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out/'T_rag_errors.csv', index=False)
    lines = [r'\begin{table}[t]', r'\centering\footnotesize',
             r'\setlength{\tabcolsep}{2pt}',
             r'\caption{FPR and FNR (\%) without and with RAG, pooled over five runs on $n$ matched policies per model and input type.}',
             r'\label{tab:rag_errors}', r'\begin{tabular}{ccccc}', r'\toprule',
             r'Model & Unit & $n$ & FPR & FNR \\', r'\midrule']
    labels = {'allam-7b':'ALLaM','llama-3.1-8b':'LLaMA','qwen-2.5-7b':'Qwen','mistral-7b':'Mistral'}
    for i, r in t.iterrows():
        if i and t.iloc[i-1].model != r.model: lines.append(r'\midrule')
        model = labels[r.model] if i == 0 or t.iloc[i-1].model != r.model else ''
        def pair(metric):
            a, b = r[metric+'_without_rag'], r[metric+'_with_rag']
            return '--' if pd.isna(a) or pd.isna(b) else f'${a:.2f}'+r'\rightarrow'+f'{b:.2f}$'
        lines.append(f'{model} & {r.unit.title()} & {r.n_policies} & {pair("fpr")} & {pair("fnr")}' + r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'T_rag_errors.tex').write_text('\n'.join(lines), encoding='utf-8')

def pooled_confusion_figure(D,out):
    cells=config_cells(D);rule=D.get('aggregation','or');GREEN,RED=np.array([57,134,90])/255,np.array([201,75,75])/255
    fig,axs=plt.subplots(4,4,figsize=(W2,7.9));export=[]
    for i,m in enumerate(MODELS):
        for j,st in enumerate(STRATS):
            ax=axs[i,j];Y,P,n,runs=cells[m,st];c=confusion_counts(Y,P);M=np.array([[c['tn'],c['fp']],[c['fn'],c['tp']]]);rs=M.sum(1,keepdims=True)
            share=np.divide(M,rs,out=np.full(M.shape,np.nan),where=rs>0);img=np.ones((2,2,3))
            for a in range(2):
                for b in range(2):
                    w=0 if np.isnan(share[a,b]) else .85*share[a,b];img[a,b]=(1-w)*np.ones(3)+w*(GREEN if a==b else RED)
            ax.imshow(img,aspect='equal')
            for a in range(2):
                for b in range(2):
                    w=0 if np.isnan(share[a,b]) else share[a,b];col='white' if .85*w>.5 else INK;lab=[['TN','FP'],['FN','TP']][a][b]
                    ax.text(b,a-.2,lab,ha='center',va='center',fontsize=6,color=col,fontweight='bold');ax.text(b,a+.02,f'{M[a,b]:,}',ha='center',va='center',fontsize=7,color=col)
                    ax.text(b,a+.26,'--' if np.isnan(share[a,b]) else f'{share[a,b]*100:.1f}%',ha='center',va='center',fontsize=5.3,color=col)
            ax.set_xticks([0,1],['Not\ndisclosed','Disclosed'],fontsize=5.3);ax.set_yticks([0,1],['Not disclosed','Disclosed'],fontsize=5.3,rotation=90,va='center')
            ax.tick_params(length=0,pad=1.5);ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True);ax.set_xlim(-.5,1.5);ax.set_ylim(1.5,-.5);ax.grid(which='minor',color='white',lw=1.4);ax.tick_params(which='minor',length=0)
            for sp in ax.spines.values():sp.set_visible(False)
            ax.set_title(f'{NAMES[m]} · {SLAB[st]}\n',fontsize=6,color=MCOL[m],fontweight='bold',loc='center',pad=9)
            ax.text(.5,1.03,f'{n} policies · {int(Y.size):,} decisions',transform=ax.transAxes,ha='center',va='bottom',fontsize=5.3,color=INK)
            if j==0:ax.set_ylabel('Expert label',fontsize=6.5,labelpad=2)
            if i==3:ax.set_xlabel('Model prediction',fontsize=6.5,labelpad=2)
            export.append(dict(model=m,strategy=st,rule=rule if st.startswith('section') else 'full',n_policies=n,n_runs=runs,n_decisions=int(Y.size),**c,
                               expert_negative=c['tn']+c['fp'],expert_positive=c['fn']+c['tp'],tn_pct_of_expert_negative=share[0,0]*100,fp_pct_of_expert_negative=share[0,1]*100,
                               fn_pct_of_expert_positive=share[1,0]*100,tp_pct_of_expert_positive=share[1,1]*100))
    fig.subplots_adjust(left=.07,right=.985,top=.93,bottom=.045,hspace=.55,wspace=.22)
    fig.legend(handles=[Patch(facecolor=GREEN,label='Correct (TN, TP)'),Patch(facecolor=RED,label='Error (FP, FN)')],loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False,fontsize=6.5)
    save(fig,out,'F7_pooled_confusion');pd.DataFrame(export).to_csv(out/'F7_pooled_confusion_values.csv',index=False)
    (out/'F7_pooled_confusion_caption.txt').write_text(
        'Pooled confusion matrices for the 16 model x strategy configurations. Rows give the expert label and columns the model prediction (positive class: disclosed); '
        'TN, FP, FN and TP are exact counts of policy x criterion decisions pooled over the five repetitions of the policies successfully evaluated in each configuration '
        '(policy coverage and decision totals are shown above each panel). Section strategies use the '+RLAB[rule]+' rule across sections. '
        'Failed records are excluded, not scored as zero. The pooled cells are repeated predictions on the same policies, and criteria within a policy may be related, so they are descriptive and not independent observations. '
        'Percentages are the share of decisions with that expert label (TN%: specificity; TP%: recall).\n',encoding='utf-8')

def make_figures(evaluation,repo='.',check_only=False,rubric=None,clean_old=False,aggregation='or',boot=1000,skip_binary=False):
    e=Path(evaluation);out=e/('figures' if aggregation=='or' else f'figures_{aggregation}');out.mkdir(parents=True,exist_ok=True);setup();D=load_all(e,repo);D['aggregation']=aggregation;E,Wn,M=validate(D)
    print(f'Validation: {len(E)} errors, {len(Wn)} warnings, {len(M)} missing inputs')
    for x in E+Wn+M:print(x)
    if E:raise SystemExit('Validation failed')
    if D['expert'] is None:raise SystemExit('Expert workbook required')
    if check_only:return
    if aggregation!='or':D=select_rule(D,aggregation,boot)
    D['grand_table'].to_csv(out/'metrics_for_selected_rule.csv',index=False)
    final_overall(D,out);final_applications(D,out);final_rubrics(D,out,binary=not skip_binary);rag_bars(D,out);aggregation_figure(D,out);stability_figure(D,out);pooled_confusion_figure(D,out);write_main_table(D,out);write_criterion_table(D,out);write_rag_errors(D,out)
    (out/'aggregation_used.txt').write_text('Section aggregation: '+aggregation+'\nRAG bars compare matching policies within each model/unit.\nAll main figures use this same fixed rule. Aggregation figure compares all four rules.\nSeparate first-run FP/FN and exact McNemar p-values (unadjusted, exploratory), Table 6 and F7 are recomputed from saved vectors under this same rule.\nNo model inference. No PDFs generated.\n',encoding='utf-8')
    print('Outputs:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Selected paper PNG figures from saved results; no model runs.')
    p.add_argument('--evaluation',required=True);p.add_argument('--repo',default='.');p.add_argument('--check-only',action='store_true');p.add_argument('--aggregation',choices=RULES,default='or');p.add_argument('--boot',type=int,default=1000);p.add_argument('--skip-binary',action='store_true',help='Keep existing binary grids; skip generating new ones');a=p.parse_args()
    if a.boot<100:p.error('--boot must be at least 100')
    make_figures(a.evaluation,a.repo,a.check_only,aggregation=a.aggregation,boot=a.boot,skip_binary=a.skip_binary)
