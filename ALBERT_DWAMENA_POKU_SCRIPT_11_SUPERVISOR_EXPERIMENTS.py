"""Additional supervisor experiments. No test-set fitting, same Script-1 held-out split.
These are diagnostic single-split comparisons, not nested-CV tuned models.
"""
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SP
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT); cfg=load_json(D['artifacts']/'script0_schema.json')
df=load_table(resolve_data_path(ROOT,cfg['student_dataset_path'])); nums=cfg['numeric_columns'];cats=cfg['categorical_columns']; y=df[cfg['target_column']].map(cfg['target_mapping']).astype(int)
split=pd.read_csv(D['artifacts']/'script1_split_indices.csv'); tr=split.train_index.dropna().astype(int).to_numpy(); te=split.test_index.dropna().astype(int).to_numpy()
seed=cfg['random_state']; REG={'reg_alpha':.1,'reg_lambda':2.,'gamma':.1,'min_child_weight':3.}
def prep(numeric):
 return ColumnTransformer([('num',SP([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',SP([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)],sparse_threshold=0)
def run(name,kind='xgb',smote=True,weighted=False,regular=False,select=False,depth=6,ratio=False):
 numeric=nums.copy(); X=df[nums+cats].copy()
 if ratio:
  # only if explicitly requested sensitivity; zero-denominator safely handled
  X['Approved / enrolled (2nd sem)']=X['Curricular units 2nd sem (approved)']/X['Curricular units 2nd sem (enrolled)'].replace(0,np.nan)
  numeric+=['Approved / enrolled (2nd sem)']
 if kind=='xgb':
  m=XGBClassifier(n_estimators=300,max_depth=depth,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='aucpr',random_state=seed,n_jobs=1,**(REG if regular else {}))
 elif kind=='lr': m=LogisticRegression(max_iter=3000,solver='lbfgs',class_weight='balanced' if weighted else None,random_state=seed)
 else: m=RandomForestClassifier(n_estimators=400,class_weight='balanced' if weighted else None,random_state=seed,n_jobs=1)
 steps=[('pre',prep(numeric))]
 if select: steps.append(('select',SelectKBest(partial(mutual_info_classif,random_state=seed),k=20)))
 if smote:steps.append(('smote',SMOTE(random_state=seed)))
 steps.append(('model',m));pipe=Pipeline(steps)
 pipe.fit(X.iloc[tr],y.iloc[tr]); prob=pipe.predict_proba(X.iloc[te])[:,1];pred=(prob>=.5).astype(int)
 tn,fp,fn,tp=confusion_matrix(y.iloc[te],pred,labels=[0,1]).ravel()
 return {'experiment':name,'model':kind,'smote':smote,'class_weight':weighted,'regularization':regular,'feature_selection':select,'depth':depth,'ratio_feature':ratio,'n_train':len(tr),'n_test':len(te),'accuracy':accuracy_score(y.iloc[te],pred),'precision':precision_score(y.iloc[te],pred,zero_division=0),'recall':recall_score(y.iloc[te],pred,zero_division=0),'f1':f1_score(y.iloc[te],pred,zero_division=0),'roc_auc':roc_auc_score(y.iloc[te],prob),'auc_pr':average_precision_score(y.iloc[te],prob),'tn':tn,'fp':fp,'fn':fn,'tp':tp}
rows=[]
experiments=[('xgb_canonical','xgb',True,False,False,False,6,False),('xgb_no_smote','xgb',False,False,False,False,6,False),('xgb_regularization_only','xgb',True,False,True,False,6,False),('xgb_feature_refinement_only','xgb',True,False,False,True,6,False),('xgb_regularization_and_refinement','xgb',True,False,True,True,6,False),('xgb_round1_depth4','xgb',True,False,False,False,4,False),('xgb_approved_enrolled_ratio','xgb',True,False,False,False,6,True)]
for kind in ('lr','rf'):
 for sm,wt,lab in [(True,False,'smote_only'),(False,True,'weights_only'),(True,True,'smote_and_weights')]:experiments.append((f'{kind}_{lab}',kind,sm,wt,False,False,6,False))
for name,kind,sm,wt,reg,sel,depth,ratio in experiments:
 print('START',name,flush=True)
 try: rows.append(run(name,kind,sm,wt,reg,sel,depth,ratio))
 except Exception as e: rows.append({'experiment':name,'error':str(e)})
 pd.DataFrame(rows).to_csv(D['results']/'script11_supervisor_experiments.csv',index=False)
print('SCRIPT 11 COMPLETE',flush=True)
