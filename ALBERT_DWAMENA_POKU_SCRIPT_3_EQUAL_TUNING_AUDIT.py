"""SCRIPT 3 — baseline fairness audit with equal GridSearchCV effort.
Refits models deliberately; output is an audit, not a replacement for Script 1 artifacts.
"""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT); cfg=load_json(D['artifacts']/'script0_schema.json'); require_no_fillins(cfg)
p=resolve_data_path(ROOT,cfg['student_dataset_path']); df=load_table(p)
TARGET=cfg['target_column']; cats=cfg.get('categorical_columns',[]); nums=cfg.get('numeric_columns',[])
y=df[TARGET].map(cfg['target_mapping']) if cfg.get('target_mapping') else df[TARGET].astype(int); X=df[nums+cats]
Xtr,_,ytr,_=train_test_split(X,y,test_size=float(cfg.get('test_size',.2)),stratify=y,random_state=int(cfg.get('random_state',42)))
pre=ColumnTransformer([('num',SkPipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),nums),('cat',SkPipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),cats)])
import os
seed=int(cfg.get('random_state',42)); folds=3 if os.environ.get('ALBERT_SMOKE_TEST')=='1' else int(cfg.get('cv_folds',10)); cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
specs={
 'logistic_regression':(LogisticRegression(max_iter=1000,solver='liblinear',class_weight='balanced',random_state=seed),{'model__C':[0.1,1.0]}),
 'decision_tree':(DecisionTreeClassifier(class_weight='balanced',random_state=seed),{'model__max_depth':[3,5]}),
 'random_forest':(RandomForestClassifier(class_weight='balanced',n_jobs=1,random_state=seed),{'model__n_estimators':[200,400]}),
 'xgboost':(XGBClassifier(subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='aucpr',random_state=seed,n_jobs=1),{'model__n_estimators':[200,300]})}
rows=[]
for n,(m,g) in specs.items():
    gs=GridSearchCV(Pipeline([('pre',pre),('smote',SMOTE(random_state=seed)),('model',m)]),g,scoring='average_precision',cv=cv,n_jobs=1,refit=False)
    gs.fit(Xtr,ytr); rows.append({'model':n,'best_cv_auc_pr':float(gs.best_score_),'best_params':json.dumps(gs.best_params_,sort_keys=True)})
pd.DataFrame(rows).to_csv(D['results']/'script3_equal_tuning_audit.csv',index=False)
print('SCRIPT 3 COMPLETE')
