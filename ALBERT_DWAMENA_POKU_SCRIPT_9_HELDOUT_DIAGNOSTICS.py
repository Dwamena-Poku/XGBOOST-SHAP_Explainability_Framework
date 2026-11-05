"""Reproducible held-out diagnostics; never refits or modifies canonical models."""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT); cfg=load_json(D['artifacts']/'script0_schema.json')
x=pd.read_csv(D['artifacts']/'script1_predictions_xgboost.csv')
rf=pd.read_csv(D['artifacts']/'script1_predictions_random_forest.csv')
lr=pd.read_csv(D['artifacts']/'script1_predictions_logistic_regression.csv')
for label,z in [('rf',rf),('lr',lr)]:
 assert np.array_equal(x.row_index,z.row_index) and np.array_equal(x.y_true,z.y_true),f'{label} prediction rows do not match'
y=x.y_true.to_numpy(dtype=int)
def met(yt,p,th=.5):
 pred=(p>=th).astype(int)
 return {'accuracy':accuracy_score(yt,pred),'precision':precision_score(yt,pred,zero_division=0),'recall':recall_score(yt,pred,zero_division=0),'f1':f1_score(yt,pred,zero_division=0),'roc_auc':roc_auc_score(yt,p) if len(np.unique(yt))>1 else np.nan,'auc_pr':average_precision_score(yt,p) if len(np.unique(yt))>1 else np.nan,'false_negatives':int(((yt==1)&(pred==0)).sum())}
rows=[]
for th in np.round(np.arange(.1,.901,.05),2):rows.append({'threshold':th,**met(y,x.y_prob.to_numpy(),th)})
pd.DataFrame(rows).to_csv(D['results']/'script9_threshold_sweep.csv',index=False)
# Paired stratified bootstrap: resample held-out cases, preserving model pairing
rng=np.random.default_rng(42);boot=[]
for comparator,frame in [('random_forest',rf),('logistic_regression',lr)]:
 metric_names=['accuracy','precision','recall','f1','roc_auc','auc_pr']
 vals={m:[] for m in metric_names}
 for _ in range(600):
  idx=rng.integers(0,len(y),len(y)); yy=y[idx]
  if len(np.unique(yy))<2:continue
  a=met(yy,x.y_prob.to_numpy()[idx]); b=met(yy,frame.y_prob.to_numpy()[idx])
  for metric in metric_names: vals[metric].append(a[metric]-b[metric])
 baseline_a=met(y,x.y_prob.to_numpy()); baseline_b=met(y,frame.y_prob.to_numpy())
 for metric in metric_names:
  orig=baseline_a[metric]-baseline_b[metric]
  boot.append({'comparison':f'xgboost_minus_{comparator}','metric':metric,'effect_absolute_difference':orig,'ci95_low':float(np.percentile(vals[metric],2.5)),'ci95_high':float(np.percentile(vals[metric],97.5)),'bootstrap_replicates':len(vals[metric])})
pd.DataFrame(boot).to_csv(D['results']/'script9_paired_bootstrap.csv',index=False)
# Subgroups: reported only when sufficient observations and both outcomes present
source=load_table(resolve_data_path(ROOT,cfg['student_dataset_path'])).iloc[x.row_index.to_numpy()]
groups={'fees':'Tuition fees up to date','scholarship':'Scholarship holder','debtor':'Debtor','gender':'Gender'}
if 'Age at enrollment' in source.columns:
 source=source.copy();source['age_band']=pd.cut(source['Age at enrollment'],bins=[0,20,25,30,40,200],labels=['<=20','21-25','26-30','31-40','41+'])
 groups['age_band']='age_band'
sub=[]
for group,col in groups.items():
 if col not in source.columns:continue
 for level,indices in source.groupby(col,observed=True).groups.items():
  ix=source.index.get_indexer(indices)
  if len(ix)<20:continue
  yy=y[ix]; p=x.y_prob.to_numpy()[ix]
  sub.append({'subgroup':group,'level':str(level),'n':len(ix),'positive_n':int(yy.sum()),**met(yy,p)})
pd.DataFrame(sub).to_csv(D['results']/'script9_subgroup_performance.csv',index=False)
miss=(y==1)&(x.y_prob.to_numpy()<.5)
miss_df=source.iloc[np.where(miss)[0]].copy();miss_df.insert(0,'row_index',x.row_index.to_numpy()[miss]);miss_df.insert(1,'predicted_dropout_probability',x.y_prob.to_numpy()[miss]);miss_df.to_csv(D['artifacts']/'script9_false_negative_case_records_RESTRICTED.csv',index=False)
summary=[]
for col in groups.values():
 if col in miss_df.columns:
  for level,n in miss_df[col].value_counts(dropna=False).items():summary.append({'feature':col,'level':str(level),'false_negative_count':int(n)})
pd.DataFrame(summary).to_csv(D['results']/'script9_false_negative_breakdown.csv',index=False)
print('Script 9 completed. FN count:',int(miss.sum()),'; review restricted case-level CSV before sharing.')
