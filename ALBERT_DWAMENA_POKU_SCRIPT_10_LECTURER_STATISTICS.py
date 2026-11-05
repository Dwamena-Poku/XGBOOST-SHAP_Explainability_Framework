"""Recalculate descriptive construct scores, Cronbach alpha, correlation and OLS regression."""
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import pearsonr
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__);D=dirs(ROOT);cfg=load_json(D['artifacts']/'script0_schema.json')
df=load_table(resolve_data_path(ROOT,cfg['teacher_dataset_path']))
constructs={'Teacher Trust':'TT','SHAP Explanation Clarity':'SH','Explanation Usability':'EU','Pedagogical Factors':'PF','Perceived Usefulness':'PU','Perceived Ease of Use':'PE','Institutional Support':'IF','Human-AI Collaboration':'HC','Prediction Quality':'PQ','Fairness and Transparency':'FT'}
scores={};stats=[]
for name,prefix in constructs.items():
 cols=[c for c in df if c.startswith(prefix) and len(c)>len(prefix) and c[len(prefix)].isdigit()]
 if not cols:raise ValueError(f'Missing items for {name}')
 a=df[cols].astype(str).apply(lambda series:pd.to_numeric(series.str.extract(r'^\s*([1-5])')[0],errors='coerce'));valid=a.dropna();k=len(cols)
 alpha=k/(k-1)*(1-valid.var(ddof=1).sum()/valid.sum(axis=1).var(ddof=1)) if k>1 else np.nan
 scores[name]=a.mean(axis=1)
 stats.append({'construct':name,'items':k,'n_complete':len(valid),'mean':float(scores[name].mean()),'sd':float(scores[name].std()),'cronbach_alpha':float(alpha)})
pd.DataFrame(stats).to_csv(D['results']/'script10_lecturer_constructs.csv',index=False)
score=pd.DataFrame(scores)
r,p=pearsonr(score['SHAP Explanation Clarity'],score['Teacher Trust'])
pd.DataFrame([{'pair':'SHAP Explanation Clarity vs Teacher Trust','r':r,'p':p,'n':len(score),'r_squared':r*r}]).to_csv(D['results']/'script10_clarity_trust_correlation.csv',index=False)
features=['Teacher Trust','SHAP Explanation Clarity','Explanation Usability','Pedagogical Factors','Perceived Usefulness','Perceived Ease of Use','Institutional Support','Prediction Quality','Fairness and Transparency']
# This is an explicit exploratory specification; do not confuse with a separately pre-registered model.
fit=sm.OLS(score['Human-AI Collaboration'],sm.add_constant(score[features]),missing='drop').fit()
pd.DataFrame({'term':fit.params.index,'coefficient':fit.params.values,'p_value':fit.pvalues.values,'t_value':fit.tvalues.values}).to_csv(D['results']/'script10_collaboration_regression.csv',index=False)
(D['results']/'script10_regression_full_summary.txt').write_text(fit.summary().as_text())
print('Script 10 complete; N=',len(score),'correlation=',r,'OLS R2=',fit.rsquared)
