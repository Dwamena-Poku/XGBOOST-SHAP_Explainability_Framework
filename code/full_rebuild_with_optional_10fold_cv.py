from pathlib import Path
import shutil,re,json,sys,platform
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import shap, nbformat as nbf
SEED=42; ROOT=Path('/mnt/data/Albert_XGBoost_SHAP_Updated_Package')
if ROOT.exists(): shutil.rmtree(ROOT)
for d in ['code','notebooks','figures/benchmark','figures/ghana','figures/lecturer','results/benchmark','results/ghana','results/lecturer','data']:(ROOT/d).mkdir(parents=True,exist_ok=True)
for s in Path('/mnt/data').glob('*'):
    if s.suffix.lower() in ['.csv','.xlsx'] and s.name not in ['MANIFEST.csv']:
        shutil.copy2(s,ROOT/'data'/s.name)
models={
'Logistic Regression':LogisticRegression(C=1,max_iter=3000,random_state=SEED),
'Decision Tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=3,random_state=SEED),
'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=1,random_state=SEED,n_jobs=1),
'XGBoost':XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=SEED,eval_metric='logloss',n_jobs=1)
}
def metrics(y,p,pr):return {'Accuracy':accuracy_score(y,p),'Precision':precision_score(y,p,zero_division=0),'Recall':recall_score(y,p,zero_division=0),'F1':f1_score(y,p,zero_division=0),'ROC_AUC':roc_auc_score(y,pr),'AUC_PR':average_precision_score(y,pr)}
def plots(df,y,probs,preds,out,prefix,fig42=False):
    ax=df.set_index('Model')[['Accuracy','Precision','Recall','F1','ROC_AUC','AUC_PR']].plot(kind='bar',figsize=(11,6));ax.set_ylim(0,1.05);ax.set_ylabel('Score');ax.set_title('Model Performance Comparison');plt.xticks(rotation=18,ha='right');plt.tight_layout();fn='Figure_4_2_model_performance_comparison.png' if fig42 else f'{prefix}model_performance_comparison.png';plt.savefig(out/fn,dpi=300,bbox_inches='tight');plt.close()
    plt.figure(figsize=(7,6))
    for n,pr in probs.items():
        fpr,tpr,_=roc_curve(y,pr);plt.plot(fpr,tpr,label=f'{n} (AUC={roc_auc_score(y,pr):.3f})')
    plt.plot([0,1],[0,1],'--');plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');plt.title('ROC Curves');plt.legend(fontsize=8);plt.tight_layout();rfn='Figure_4_3_ROC_curves.png' if fig42 else f'{prefix}ROC_curves.png';plt.savefig(out/rfn,dpi=300,bbox_inches='tight');plt.close()
    plt.figure(figsize=(7,6)); prev=np.mean(y)
    for n,pr in probs.items():
        pre,rec,_=precision_recall_curve(y,pr);plt.plot(rec,pre,label=f'{n} (AP={average_precision_score(y,pr):.3f})')
    plt.axhline(prev,linestyle='--',label=f'Prevalence={prev:.3f}');plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Precision–Recall Curves');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(out/f'{prefix}Precision_Recall_curves.png',dpi=300,bbox_inches='tight');plt.close()
    for n,p in preds.items():
        fig,ax=plt.subplots(figsize=(5,4.5));ConfusionMatrixDisplay.from_predictions(y,p,display_labels=['No intervention','Intervention'],ax=ax,colorbar=False);ax.set_title(f'{n} Confusion Matrix');plt.tight_layout();safe=re.sub('[^a-z0-9]+','_',n.lower()).strip('_');plt.savefig(out/f'{prefix}confusion_matrix_{safe}.png',dpi=300,bbox_inches='tight');plt.close()
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for ax,(n,p) in zip(axs.flat,preds.items()):ConfusionMatrixDisplay.from_predictions(y,p,display_labels=['No','Intervention'],ax=ax,colorbar=False);ax.set_title(n)
    plt.tight_layout();plt.savefig(out/f'{prefix}confusion_matrices_all_models.png',dpi=300,bbox_inches='tight');plt.close()
def shapplots(model,X,features,out,prefix,resdir):
    ex=shap.TreeExplainer(model);sv=ex(X);v=np.asarray(sv.values);v=v[:,:,1] if v.ndim==3 else v
    pd.DataFrame({'Feature':features,'MeanAbsSHAP':np.abs(v).mean(0)}).sort_values('MeanAbsSHAP',ascending=False).to_csv(resdir/f'{prefix}shap_global_feature_ranking.csv',index=False)
    shap.summary_plot(v,X,feature_names=features,plot_type='bar',show=False,max_display=15);plt.tight_layout();plt.savefig(out/f'{prefix}SHAP_global_bar.png',dpi=300,bbox_inches='tight');plt.close()
    shap.summary_plot(v,X,feature_names=features,show=False,max_display=15);plt.tight_layout();plt.savefig(out/f'{prefix}SHAP_beeswarm.png',dpi=300,bbox_inches='tight');plt.close()
    bases=np.ravel(sv.base_values)
    for i in range(min(3,len(X))):
        base=bases[i] if len(bases)==len(X) else bases[0]
        e=shap.Explanation(values=v[i],base_values=base,data=np.asarray(X)[i],feature_names=features);shap.plots.waterfall(e,max_display=12,show=False);plt.tight_layout();plt.savefig(out/f'{prefix}SHAP_waterfall_case_{i+1}.png',dpi=300,bbox_inches='tight');plt.close()
# benchmark
b=pd.read_csv('/mnt/data/data(3).csv',sep=';');b.columns=[c.strip().strip('"') for c in b.columns];yb=(b.Target.astype(str).str.strip()=='Dropout').astype(int);Xb=b.drop(columns='Target').apply(pd.to_numeric,errors='coerce');Xb=Xb.fillna(Xb.median())
Xtr,Xte,ytr,yte=train_test_split(Xb,yb,test_size=.2,stratify=yb,random_state=SEED);sm=SMOTE(random_state=SEED);Xres,yres=sm.fit_resample(Xtr,ytr);sc=StandardScaler();XresS=sc.fit_transform(Xres);XteS=sc.transform(Xte)
rows=[];probs={};preds={};fitted={}
for n,m in models.items():
    m.fit(XresS,yres);pr=m.predict_proba(XteS)[:,1];p=(pr>=.5).astype(int);rows.append({'Model':n,**metrics(yte,p,pr)});probs[n]=pr;preds[n]=p;fitted[n]=m
bd=pd.DataFrame(rows);bd.to_csv(ROOT/'results/benchmark/heldout_test_metrics.csv',index=False);plots(bd,yte,probs,preds,ROOT/'figures/benchmark','benchmark_',True);shapplots(fitted['XGBoost'],XteS,list(Xb.columns),ROOT/'figures/benchmark','benchmark_',ROOT/'results/benchmark')
# 10fold summary fixed params
cv=StratifiedKFold(10,shuffle=True,random_state=SEED);cvrows=[]
for n,m in models.items():
    pipe=ImbPipeline([('scale',StandardScaler()),('smote',SMOTE(random_state=SEED)),('model',m)])
    for metric,scor in [('ROC_AUC','roc_auc'),('AUC_PR','average_precision'),('F1','f1')]:
        scs=cross_val_score(pipe,Xb,yb,cv=cv,scoring=scor,n_jobs=1);cvrows.append({'Model':n,'Metric':metric,'Mean':scs.mean(),'SD':scs.std(ddof=1)})
pd.DataFrame(cvrows).to_csv(ROOT/'results/benchmark/cross_validation_summary.csv',index=False)
# ghana
g=[]
for f in Path('/mnt/data').glob('*gpa*csv'):
    d=pd.read_csv(f);d['CourseCode']=f.name.split('+')[0];g.append(d)
gh=pd.concat(g,ignore_index=True)
for c in ['10(10)','10(10).1','20(20)','EXAM(60)']:gh[c]=pd.to_numeric(gh[c],errors='coerce')
gh['OverallScore']=gh[['10(10)','10(10).1','20(20)','EXAM(60)']].sum(axis=1,min_count=4);gc=gh.dropna(subset=['OverallScore','20(20)']).copy();gc['Intervention']=(gc.OverallScore<50).astype(int)
Xg=gc[['20(20)','CourseCode']];yg=gc.Intervention;pre=ColumnTransformer([('num',StandardScaler(),['20(20)']),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['CourseCode'])]);Xt,Xv,yt,yv=train_test_split(Xg,yg,test_size=.2,stratify=yg,random_state=SEED);A=pre.fit_transform(Xt);B=pre.transform(Xv);smg=SMOTE(random_state=SEED);Ar,yr=smg.fit_resample(A,yt)
rows=[];probs={};preds={};gf={}
for n,m in models.items():
    # clone fresh by constructor via sklearn clone
    from sklearn.base import clone
    mm=clone(m);mm.fit(Ar,yr);pr=mm.predict_proba(B)[:,1];p=(pr>=.5).astype(int);rows.append({'Model':n,**metrics(yv,p,pr)});probs[n]=pr;preds[n]=p;gf[n]=mm
gd=pd.DataFrame(rows);gd.to_csv(ROOT/'results/ghana/heldout_test_metrics.csv',index=False);plots(gd,yv,probs,preds,ROOT/'figures/ghana','ghana_',False);shapplots(gf['XGBoost'],B,list(pre.get_feature_names_out()),ROOT/'figures/ghana','ghana_',ROOT/'results/ghana')
pd.DataFrame({'Measure':['Initial records','Complete cases','Intervention cases','Non-intervention cases','Intervention prevalence','Overall mean','Overall SD'],'Value':[len(gh),len(gc),int(yg.sum()),int((1-yg).sum()),yg.mean(),gc.OverallScore.mean(),gc.OverallScore.std(ddof=1)]}).to_csv(ROOT/'results/ghana/dataset_summary.csv',index=False)
# lecturer descriptive
l=pd.read_csv('/mnt/data/Teacher_Trust_Human_AI_dataset.csv');cons={'Teacher Trust':'TT','SHAP Explanation Clarity':'SH','Explanation Usability':'EU','Pedagogical Factors':'PF','Perceived Usefulness':'PU','Perceived Ease of Use':'PE','Institutional Support':'IF','Human-AI Collaboration':'HC','Prediction Quality':'PQ','Fairness and Transparency':'FT'}
def lik(s):return pd.to_numeric(s.astype(str).str.extract(r'([1-5])')[0],errors='coerce')
def alpha(x):
    x=x.dropna();k=x.shape[1];return k/(k-1)*(1-x.var(ddof=1).sum()/x.sum(1).var(ddof=1))
rr=[]
for name,pfx in cons.items():
    cs=[c for c in l.columns if re.match(rf'^{pfx}\d+:',c)];x=pd.DataFrame({c:lik(l[c]) for c in cs});s=x.mean(1);rr.append({'Construct':name,'Items':len(cs),'N':s.notna().sum(),'Mean':s.mean(),'SD':s.std(ddof=1),'Minimum':s.min(),'Maximum':s.max(),'Cronbach_alpha':alpha(x)})
ld=pd.DataFrame(rr);ld.to_csv(ROOT/'results/lecturer/table_4_5_construct_statistics.csv',index=False);l['Record Status'].value_counts().rename_axis('Record Status').reset_index(name='Count').to_csv(ROOT/'results/lecturer/record_status_counts.csv',index=False)
ax=ld.set_index('Construct').Mean.sort_values().plot(kind='barh',figsize=(9,6));ax.set_xlabel('Mean (1–5)');plt.tight_layout();plt.savefig(ROOT/'figures/lecturer/lecturer_construct_means.png',dpi=300,bbox_inches='tight');plt.close()
# scripts/notebook/readme
shutil.copy2('/mnt/data/fast_build.py',ROOT/'code/rebuild_all_analysis.py')
nb=nbf.v4.new_notebook();nb.cells=[nbf.v4.new_markdown_cell('# Canonical XGBoost–SHAP Analysis'),nbf.v4.new_code_cell('%run ../code/rebuild_all_analysis.py')];nbf.write(nb,ROOT/'notebooks/Updated_XGBoost_SHAP_Analysis.ipynb')
(ROOT/'requirements.txt').write_text(Path('/mnt/data/requirements.txt').read_text())
(ROOT/'README.md').write_text('''# Updated XGBoost–SHAP Reproducibility Package\n\nAll requested figures are regenerated from a single canonical run. Figures are saved as 300-dpi PNG files.\n\n## Manuscript figures\n- Figure 4.2: `figures/benchmark/Figure_4_2_model_performance_comparison.png`\n- Figure 4.3: `figures/benchmark/Figure_4_3_ROC_curves.png`\n- Precision–Recall curves, individual and combined confusion matrices, and SHAP global/beeswarm/waterfall figures are in `figures/benchmark/`.\n- Ghanaian institutional equivalents are in `figures/ghana/`.\n\n## Canonical settings\nRandom seed 42; stratified 80:20 split; SMOTE applied only to training data; fixed tuned model parameters; benchmark 10-fold CV summary included.\n\n## Ghanaian early-warning model\nIntervention is defined as OverallScore < 50. EXAM(60) and OverallScore are excluded from inputs to prevent target leakage.\n\n## Lecturer data\nThe supplied 350-row file is analysed separately. Check `results/lecturer/record_status_counts.csv` before making collection-status claims.\n''')
# manifest + zip
items=[]
for p in ROOT.rglob('*'):
    if p.is_file():items.append({'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size})
pd.DataFrame(items).sort_values('path').to_csv(ROOT/'MANIFEST.csv',index=False)
print('BENCH\n',bd.to_string(index=False));print('GHANA\n',gd.to_string(index=False));print('files',len(items))
