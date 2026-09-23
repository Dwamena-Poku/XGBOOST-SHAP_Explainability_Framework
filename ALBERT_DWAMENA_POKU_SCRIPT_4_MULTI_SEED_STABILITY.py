"""SCRIPT 4 — 5-seed stability audit with variance; deliberately refits."""
import pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT); cfg=load_json(D['artifacts']/'script0_schema.json'); require_no_fillins(cfg)
p=resolve_data_path(ROOT,cfg['student_dataset_path']); df=load_table(p)
nums=cfg.get('numeric_columns',[]); cats=cfg.get('categorical_columns',[]); X=df[nums+cats]; y=df[cfg['target_column']].map(cfg['target_mapping']) if cfg.get('target_mapping') else df[cfg['target_column']].astype(int)
base=joblib.load(D['artifacts']/'script1_model_xgboost.joblib'); rows=[]
for seed in [11,23,42,71,101]:
    tr,te=train_test_split(np.arange(len(df)),test_size=float(cfg.get('test_size',.2)),stratify=y,random_state=seed)
    m=clone(base); m.set_params(model__random_state=seed); m.fit(X.iloc[tr],y.iloc[tr]); pred=m.predict(X.iloc[te]); prob=m.predict_proba(X.iloc[te])[:,1]
    rows.append({'seed':seed,'accuracy':accuracy_score(y.iloc[te],pred),'precision':precision_score(y.iloc[te],pred,zero_division=0),'recall':recall_score(y.iloc[te],pred,zero_division=0),'f1':f1_score(y.iloc[te],pred,zero_division=0),'roc_auc':roc_auc_score(y.iloc[te],prob),'auc_pr':average_precision_score(y.iloc[te],prob)})
r=pd.DataFrame(rows); r.to_csv(D['results']/'script4_multi_seed_scores.csv',index=False); r.describe().T.to_csv(D['results']/'script4_multi_seed_summary.csv')
print('SCRIPT 4 COMPLETE')
