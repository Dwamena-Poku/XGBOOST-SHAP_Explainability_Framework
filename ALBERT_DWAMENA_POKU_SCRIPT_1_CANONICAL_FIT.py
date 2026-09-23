"""SCRIPT 1 — canonical fit and artifact producer.
ANCHOR: REM-ALBERT_DWAMENA_POKU-2026-08-26
Writes fold scores as plain Python floats (logic bug fixed).
"""
from pathlib import Path
import json, joblib, platform
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from ALBERT_DWAMENA_POKU_COMMON import *

ROOT=root_from_script(__file__); D=dirs(ROOT)
SCHEMA_PATH=D['artifacts']/'script0_schema.json'
if not SCHEMA_PATH.exists():
    raise FileNotFoundError(f'Missing {SCHEMA_PATH}. Run Script 0 first.')
cfg=load_json(SCHEMA_PATH); require_no_fillins(cfg, 'Script 0 schema')
DATA_PATH=resolve_data_path(ROOT,cfg['student_dataset_path'])
TARGET=cfg['target_column']; POSITIVE=cfg.get('positive_class',1)
ID_COLS=cfg.get('id_columns',[]); CAT_COLS=cfg.get('categorical_columns',[]); NUM_COLS=cfg.get('numeric_columns',[])
TEST_SIZE=float(cfg.get('test_size',0.20)); RANDOM_STATE=int(cfg.get('random_state',42)); import os
N_SPLITS=3 if os.environ.get('ALBERT_SMOKE_TEST')=='1' else int(cfg.get('cv_folds',10))

df=load_table(DATA_PATH)
missing=[c for c in [TARGET,*CAT_COLS,*NUM_COLS] if c not in df.columns]
if missing: raise KeyError(f'Script 1 missing declared columns: {missing}. Refusing to guess headers.')
y_raw=df[TARGET]
if cfg.get('target_mapping'):
    y=y_raw.map(cfg['target_mapping'])
    if y.isna().any(): raise ValueError('target_mapping left unmapped target values')
else: y=(y_raw==POSITIVE).astype(int) if set(pd.unique(y_raw.dropna()))!={0,1} else y_raw.astype(int)
X=df[[*NUM_COLS,*CAT_COLS]].copy()
idx=np.arange(len(df)); tr_idx,te_idx=train_test_split(idx,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
Xtr,Xte=X.iloc[tr_idx],X.iloc[te_idx]; ytr,yte=y.iloc[tr_idx],y.iloc[te_idx]

num=SkPipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())])
cat=SkPipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre=ColumnTransformer([('num',num,NUM_COLS),('cat',cat,CAT_COLS)], remainder='drop', verbose_feature_names_out=True)
models={
 'logistic_regression':LogisticRegression(max_iter=3000,class_weight='balanced',random_state=RANDOM_STATE),
 'decision_tree':DecisionTreeClassifier(class_weight='balanced',random_state=RANDOM_STATE),
 'random_forest':RandomForestClassifier(n_estimators=400,class_weight='balanced',n_jobs=-1,random_state=RANDOM_STATE),
 'xgboost':XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='aucpr',random_state=RANDOM_STATE,n_jobs=-1)
}
scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc','auc_pr':'average_precision'}
cv=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
fold_rows=[]; metric_rows=[]
for name,model in models.items():
    pipe=Pipeline([('pre',pre),('smote',SMOTE(random_state=RANDOM_STATE)),('model',model)])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
    joblib.dump(pipe,D['artifacts']/f'script1_model_{name}.joblib')
    pd.DataFrame({'row_index':te_idx,'y_true':yte.to_numpy(),'y_pred':pred,'y_prob':prob}).to_csv(D['artifacts']/f'script1_predictions_{name}.csv',index=False)
    metric_rows.append({'model':name,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,prob),'auc_pr':average_precision_score(yte,prob)})
    cvres=cross_validate(pipe,Xtr,ytr,cv=cv,scoring=scoring,n_jobs=-1,return_train_score=False)
    row={'model':name}
    for m in scoring:
        vals=[float(v) for v in cvres[f'test_{m}']] # FIX: plain floats, never numpy scalar repr
        row[f'{m}_fold_scores']=json.dumps(vals)
        row[f'{m}_mean']=float(np.mean(vals)); row[f'{m}_sd']=float(np.std(vals,ddof=1))
    fold_rows.append(row)

pd.DataFrame(metric_rows).to_csv(D['results']/'script1_heldout_metrics.csv',index=False)
pd.DataFrame(fold_rows).to_csv(D['results']/'script1_fold_scores.csv',index=False)
pd.DataFrame({'train_index':pd.Series(tr_idx),'test_index':pd.Series(te_idx)}).to_csv(D['artifacts']/'script1_split_indices.csv',index=False)
# Save transformed names from fitted XGBoost pipeline; categorical values are one-hot encoded, never ordinal magnitudes.
xgb=joblib.load(D['artifacts']/'script1_model_xgboost.joblib')
feature_names=[str(x) for x in xgb.named_steps['pre'].get_feature_names_out()]
save_json({'feature_names':feature_names,'target':TARGET,'n_train':len(tr_idx),'n_test':len(te_idx),'seed':RANDOM_STATE,'cv_folds':N_SPLITS,'dataset_sha256':sha256(DATA_PATH)},D['artifacts']/'script1_contract.json')
print('SCRIPT 1 COMPLETE:', D['artifacts'])
