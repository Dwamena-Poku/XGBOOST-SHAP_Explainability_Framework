"""SCRIPT 8 — Nadeau–Bengio corrected fold tests + Holm family correction.
CONSUMES Script 1 fold-score artifact; REFITS NOTHING.
Parses both fixed plain-float lists and legacy np.float64(...) lists; errors on unparseable data.
"""
import itertools, math
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT)
fs=pd.read_csv(D['results']/'script1_fold_scores.csv')
METRICS=['accuracy','precision','recall','f1','roc_auc','auc_pr']
TEST_SIZE=None
cfg=load_json(D['artifacts']/'script0_schema.json'); test_size=float(cfg.get('test_size',.20)); train_size=1-test_size
rows=[]
for metric in METRICS:
    col=f'{metric}_fold_scores'
    if col not in fs.columns: raise KeyError(f'Unparseable/missing fold-score column: {col}')
    parsed={}
    for _,r in fs.iterrows():
        try: parsed[r['model']]=parse_score_list(r[col])
        except Exception as e: raise ValueError(f'Unparseable column {col} for model {r["model"]}: {e}') from e
    for a,b in itertools.combinations(parsed,2):
        x=np.asarray(parsed[a],float); y=np.asarray(parsed[b],float)
        if len(x)!=len(y) or len(x)<2: raise ValueError(f'Fold count mismatch for {a} vs {b}, metric={metric}')
        d=x-y; k=len(d); mean=float(d.mean()); s2=float(d.var(ddof=1))
        # Nadeau-Bengio corrected resampled t-test variance: (1/k + n_test/n_train)*s_d^2
        correction=(1/k)+(test_size/train_size)
        se=math.sqrt(correction*s2) if s2>0 else 0.0
        t=mean/se if se>0 else (np.inf if mean!=0 else 0.0)
        p=2*stats.t.sf(abs(t),df=k-1) if np.isfinite(t) else 0.0
        naive=stats.ttest_rel(x,y).pvalue
        rows.append({'metric':metric,'model_a':a,'model_b':b,'mean_difference':mean,'naive_p':float(naive),'nadeau_bengio_t':float(t),'nadeau_bengio_p':float(p),'folds':k,'correction_factor':float(correction)})
r=pd.DataFrame(rows)
if len(r):
    rej,pholm,_,_=multipletests(r['nadeau_bengio_p'].to_numpy(),method='holm'); r['holm_p']=pholm; r['holm_reject_0_05']=rej
r.to_csv(D['results']/'script8_dependence_corrected_tests.csv',index=False)
print('SCRIPT 8 COMPLETE — no refit performed. Dependence-corrected inference is authoritative for fold comparisons.')
