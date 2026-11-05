"""Fresh five-seed refit for all four models; avoids loading version-incompatible serialized estimators."""
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SP
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__);D=dirs(ROOT);cfg=load_json(D['artifacts']/'script0_schema.json');df=load_table(resolve_data_path(ROOT,cfg['student_dataset_path']))
nums=cfg['numeric_columns'];cats=cfg['categorical_columns'];X=df[nums+cats];y=df[cfg['target_column']].map(cfg['target_mapping']).astype(int)
rows=[]
for seed in [11,23,42,71,101]:
 tr,te=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=seed)
 pre=ColumnTransformer([('num',SP([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',SP([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)],sparse_threshold=0)
 models={'logistic_regression':LogisticRegression(max_iter=3000,class_weight='balanced',random_state=seed),'decision_tree':DecisionTreeClassifier(class_weight='balanced',random_state=seed),'random_forest':RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=seed,n_jobs=1),'xgboost':XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='aucpr',random_state=seed,n_jobs=1)}
 for name,m in models.items():
  print('START',seed,name,flush=True)
  p=Pipeline([('pre',pre),('smote',SMOTE(random_state=seed)),('model',m)]);p.fit(X.iloc[tr],y.iloc[tr]);pred=p.predict(X.iloc[te]);prob=p.predict_proba(X.iloc[te])[:,1]
  rows.append({'model':name,'seed':seed,'accuracy':accuracy_score(y.iloc[te],pred),'precision':precision_score(y.iloc[te],pred,zero_division=0),'recall':recall_score(y.iloc[te],pred,zero_division=0),'f1':f1_score(y.iloc[te],pred,zero_division=0),'roc_auc':roc_auc_score(y.iloc[te],prob),'auc_pr':average_precision_score(y.iloc[te],prob)})
  pd.DataFrame(rows).to_csv(D['results']/'script4_multi_seed_scores.csv',index=False)
r=pd.DataFrame(rows);r.groupby('model').agg({c:['mean','std'] for c in ['accuracy','precision','recall','f1','roc_auc','auc_pr']}).to_csv(D['results']/'script4_multi_seed_summary.csv');print('SCRIPT 4 COMPLETE')
