"""SCRIPT 6 — Phase 3 engineered-XGBoost ablation.

Final operational specification for the requested full-chain rerun:
- training-fold preprocessing only;
- SMOTE on training data only;
- feature refinement with SelectKBest(mutual_info_classif, k=20);
- explicit XGBoost regularisation parameters shown below;
- same locked 80:20 split as Script 1;
- 10-fold CV on the Script 1 training partition.

This is a newly explicit Phase-3 specification for this rerun; it must not be
misdescribed as having been part of an earlier approved methodology.
"""
import json, joblib
import numpy as np, pandas as pd
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from ALBERT_DWAMENA_POKU_COMMON import *

ROOT=root_from_script(__file__); D=dirs(ROOT)
cfg=load_json(D['artifacts']/'script0_schema.json'); require_no_fillins(cfg,'Script 0 schema')
REG_PARAMS={'reg_alpha':0.1,'reg_lambda':2.0,'gamma':0.1,'min_child_weight':3.0}
FEATURE_REFINEMENT_K=20
seed=int(cfg.get('random_state',42))

# Reuse Script 1's exact split indices; do not create a new held-out partition.
split=pd.read_csv(D['artifacts']/'script1_split_indices.csv')
tr_idx=split['train_index'].dropna().astype(int).to_numpy(); te_idx=split['test_index'].dropna().astype(int).to_numpy()
df=load_table(resolve_data_path(ROOT,cfg['student_dataset_path']))
TARGET=cfg['target_column']; cats=cfg.get('categorical_columns',[]); nums=cfg.get('numeric_columns',[])
y=df[TARGET].map(cfg['target_mapping']) if cfg.get('target_mapping') else df[TARGET].astype(int)
if y.isna().any(): raise ValueError('Phase 3 target mapping left unmapped values')
X=df[nums+cats].copy(); Xtr,Xte=X.iloc[tr_idx],X.iloc[te_idx]; ytr,yte=y.iloc[tr_idx],y.iloc[te_idx]

num=SkPipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())])
cat=SkPipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
pre=ColumnTransformer([('num',num,nums),('cat',cat,cats)],remainder='drop',verbose_feature_names_out=True)
mi=partial(mutual_info_classif, random_state=seed)
model=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,
                    objective='binary:logistic',eval_metric='aucpr',random_state=seed,n_jobs=-1,**REG_PARAMS)
pipe=Pipeline([('pre',pre),('select',SelectKBest(score_func=mi,k=FEATURE_REFINEMENT_K)),('smote',SMOTE(random_state=seed)),('model',model)])
pipe.fit(Xtr,ytr)
pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
metrics={'model':'engineered_xgboost','accuracy':float(accuracy_score(yte,pred)),'precision':float(precision_score(yte,pred,zero_division=0)),
         'recall':float(recall_score(yte,pred,zero_division=0)),'f1':float(f1_score(yte,pred,zero_division=0)),
         'roc_auc':float(roc_auc_score(yte,prob)),'auc_pr':float(average_precision_score(yte,prob))}
pd.DataFrame([metrics]).to_csv(D['results']/'script6_phase3_heldout_metrics.csv',index=False)
pd.DataFrame({'row_index':te_idx,'y_true':yte.to_numpy(),'y_pred':pred,'y_prob':prob}).to_csv(D['artifacts']/'script6_phase3_predictions.csv',index=False)
joblib.dump(pipe,D['artifacts']/'script6_model_engineered_xgboost.joblib')

tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
pd.DataFrame([{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}]).to_csv(D['results']/'script6_phase3_confusion_counts.csv',index=False)

cv=StratifiedKFold(n_splits=int(cfg.get('cv_folds',10)),shuffle=True,random_state=seed)
scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc','auc_pr':'average_precision'}
cvres=cross_validate(pipe,Xtr,ytr,cv=cv,scoring=scoring,n_jobs=-1,return_train_score=False)
row={'model':'engineered_xgboost'}
for m in scoring:
    vals=[float(v) for v in cvres[f'test_{m}']]
    row[f'{m}_fold_scores']=json.dumps(vals); row[f'{m}_mean']=float(np.mean(vals)); row[f'{m}_sd']=float(np.std(vals,ddof=1))
pd.DataFrame([row]).to_csv(D['results']/'script6_phase3_fold_scores.csv',index=False)

# Audit selected transformed features from the final held-out fit.
feature_names=np.asarray(pipe.named_steps['pre'].get_feature_names_out(),dtype=object)
mask=pipe.named_steps['select'].get_support()
pd.DataFrame({'selected_feature':feature_names[mask]}).to_csv(D['results']/'script6_phase3_selected_features.csv',index=False)
status={'arm':'Phase 3 engineered XGBoost','status':'COMPLETE','REG_PARAMS':json.dumps(REG_PARAMS,sort_keys=True),
        'FEATURE_REFINEMENT_K':FEATURE_REFINEMENT_K,'split_source':'script1_split_indices.csv','n_train':len(tr_idx),'n_test':len(te_idx)}
pd.DataFrame([status]).to_csv(D['results']/'script6_phase3_status.csv',index=False)
print('SCRIPT 6 COMPLETE:', metrics)
