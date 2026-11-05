"""Read-only consistency checks; does not claim to reproduce expensive training."""
from pathlib import Path
import json, hashlib, pandas as pd, numpy as np
R=Path(__file__).resolve().parent
checks=[]
def record(name,ok,detail): checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})
for fn in ['ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv','ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv','ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv']:
 p=R/'data'/fn; record('dataset '+fn,p.exists(), hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'missing')
for model in ['logistic_regression','decision_tree','random_forest','xgboost']:
 p=R/'artifacts'/f'script1_predictions_{model}.csv';record('predictions '+model,p.exists(),len(pd.read_csv(p)) if p.exists() else 'missing')
a=pd.read_csv(R/'results/script3_equal_tuning_audit.csv');record('tuning ten folds',len(a)==4 and (a.cv_folds==10).all() and (~a.smoke_test.astype(bool)).all(),a[['model','best_cv_auc_pr']].to_dict('records'))
s=pd.read_csv(R/'results/script7_shap_ranking_source_features.csv');record('SHAP ranking',s.iloc[0].source_feature=='Curricular units 2nd sem (approved)' and s.iloc[1].source_feature=='Tuition fees up to date',s.head(5).to_dict('records'))
b=pd.read_csv(R/'results/script8_dependence_corrected_tests.csv');record('NB 10-fold factor',np.allclose(b.correction_factor,1/10+1/9),b.correction_factor.iloc[0]);record('Holm family',len(b)==36,len(b))
ph=pd.read_csv(R/'results/script6_phase3_heldout_metrics.csv').iloc[0]; cm=pd.read_csv(R/'results/script6_phase3_confusion_counts.csv').iloc[0];record('phase3 metric/confusion consistency',int(cm.tn+cm.fp+cm.fn+cm.tp)==885 and abs(ph.accuracy-(cm.tn+cm.tp)/885)<1e-8,ph.to_dict())
q=pd.read_csv(R/'results/script10_clarity_trust_correlation.csv').iloc[0];record('lecturer correlation',q.n==350 and abs(q.r-.61283)<.0001,q.to_dict())
r=pd.read_csv(R/'results/script10_collaboration_regression.csv');record('lecturer predictor count',len(r)==10,'9 predictors + intercept; thesis 8-predictor specification differs')
pd.DataFrame(checks).to_csv(R/'results/verification_release_checks.csv',index=False)
print(json.dumps({'passed':sum(c['passed'] for c in checks),'total':len(checks),'note':'Checks verify stored-file consistency, not end-to-end model reproducibility'},indent=2))
