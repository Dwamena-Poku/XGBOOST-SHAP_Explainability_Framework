from pathlib import Path
import shutil,re,json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap, nbformat as nbf
ROOT=Path('/mnt/data/Albert_XGBoost_SHAP_Updated_Package'); SEED=42
models={'Logistic Regression':LogisticRegression(C=1,max_iter=3000,random_state=SEED),'Decision Tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=3,random_state=SEED),'Random Forest':RandomForestClassifier(n_estimators=120,max_depth=8,random_state=SEED,n_jobs=1),'XGBoost':XGBClassifier(n_estimators=120,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=SEED,eval_metric='logloss',n_jobs=1)}
def mets(y,p,pr):return {'Accuracy':accuracy_score(y,p),'Precision':precision_score(y,p,zero_division=0),'Recall':recall_score(y,p,zero_division=0),'F1':f1_score(y,p,zero_division=0),'ROC_AUC':roc_auc_score(y,pr),'AUC_PR':average_precision_score(y,pr)}
def plotset(df,y,probs,preds,out,prefix):
    ax=df.set_index('Model')[['Accuracy','Precision','Recall','F1','ROC_AUC','AUC_PR']].plot(kind='bar',figsize=(11,6));ax.set_ylim(0,1.05);ax.set_ylabel('Score');ax.set_title('Ghanaian Institutional Model Performance');plt.xticks(rotation=18,ha='right');plt.tight_layout();plt.savefig(out/f'{prefix}model_performance_comparison.png',dpi=300,bbox_inches='tight');plt.close()
    plt.figure(figsize=(7,6))
    for n,pr in probs.items():fpr,tpr,_=roc_curve(y,pr);plt.plot(fpr,tpr,label=f'{n} (AUC={roc_auc_score(y,pr):.3f})')
    plt.plot([0,1],[0,1],'--');plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');plt.title('Ghanaian ROC Curves');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(out/f'{prefix}ROC_curves.png',dpi=300,bbox_inches='tight');plt.close()
    plt.figure(figsize=(7,6));prev=np.mean(y)
    for n,pr in probs.items():pre,rec,_=precision_recall_curve(y,pr);plt.plot(rec,pre,label=f'{n} (AP={average_precision_score(y,pr):.3f})')
    plt.axhline(prev,linestyle='--',label=f'Prevalence={prev:.3f}');plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Ghanaian Precision–Recall Curves');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(out/f'{prefix}Precision_Recall_curves.png',dpi=300,bbox_inches='tight');plt.close()
    for n,p in preds.items():
        fig,ax=plt.subplots(figsize=(5,4.5));ConfusionMatrixDisplay.from_predictions(y,p,display_labels=['No intervention','Intervention'],ax=ax,colorbar=False);ax.set_title(f'{n} Confusion Matrix');plt.tight_layout();safe=re.sub('[^a-z0-9]+','_',n.lower()).strip('_');plt.savefig(out/f'{prefix}confusion_matrix_{safe}.png',dpi=300,bbox_inches='tight');plt.close()
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for ax,(n,p) in zip(axs.flat,preds.items()):ConfusionMatrixDisplay.from_predictions(y,p,display_labels=['No','Intervention'],ax=ax,colorbar=False);ax.set_title(n)
    plt.tight_layout();plt.savefig(out/f'{prefix}confusion_matrices_all_models.png',dpi=300,bbox_inches='tight');plt.close()
# Ghana
frames=[]
for f in Path('/mnt/data').glob('*gpa*csv'):
    d=pd.read_csv(f);d['CourseCode']=f.name.split('+')[0];frames.append(d)
gh=pd.concat(frames,ignore_index=True)
for c in ['10(10)','10(10).1','20(20)','EXAM(60)']:gh[c]=pd.to_numeric(gh[c],errors='coerce')
gh['OverallScore']=gh[['10(10)','10(10).1','20(20)','EXAM(60)']].sum(axis=1,min_count=4);gc=gh.dropna(subset=['OverallScore','20(20)']).copy();gc['Intervention']=(gc.OverallScore<50).astype(int)
X=gc[['20(20)','CourseCode']];y=gc.Intervention;pre=ColumnTransformer([('num',StandardScaler(),['20(20)']),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['CourseCode'])]);Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED);A=pre.fit_transform(Xt);B=pre.transform(Xv);Ar,yr=SMOTE(random_state=SEED).fit_resample(A,yt)
rows=[];probs={};preds={};fits={}
for n,m in models.items():mm=clone(m);mm.fit(Ar,yr);pr=mm.predict_proba(B)[:,1];p=(pr>=.5).astype(int);rows.append({'Model':n,**mets(yv,p,pr)});probs[n]=pr;preds[n]=p;fits[n]=mm
gd=pd.DataFrame(rows);gd.to_csv(ROOT/'results/ghana/heldout_test_metrics.csv',index=False);plotset(gd,yv,probs,preds,ROOT/'figures/ghana','ghana_')
pd.DataFrame({'Measure':['Initial records','Complete cases','Intervention cases','Non-intervention cases','Intervention prevalence','Overall mean','Overall SD'],'Value':[len(gh),len(gc),int(y.sum()),int((1-y).sum()),y.mean(),gc.OverallScore.mean(),gc.OverallScore.std(ddof=1)]}).to_csv(ROOT/'results/ghana/dataset_summary.csv',index=False)
# Ghana SHAP
m=fits['XGBoost'];ex=shap.TreeExplainer(m);sv=ex(B);v=np.asarray(sv.values);v=v[:,:,1] if v.ndim==3 else v;names=list(pre.get_feature_names_out());pd.DataFrame({'Feature':names,'MeanAbsSHAP':np.abs(v).mean(0)}).sort_values('MeanAbsSHAP',ascending=False).to_csv(ROOT/'results/ghana/ghana_shap_global_feature_ranking.csv',index=False)
shap.summary_plot(v,B,feature_names=names,plot_type='bar',show=False,max_display=15);plt.tight_layout();plt.savefig(ROOT/'figures/ghana/ghana_SHAP_global_bar.png',dpi=300,bbox_inches='tight');plt.close();shap.summary_plot(v,B,feature_names=names,show=False,max_display=15);plt.tight_layout();plt.savefig(ROOT/'figures/ghana/ghana_SHAP_beeswarm.png',dpi=300,bbox_inches='tight');plt.close()
bases=np.ravel(sv.base_values)
for i in range(3):
    base=bases[i] if len(bases)==len(B) else bases[0];e=shap.Explanation(values=v[i],base_values=base,data=np.asarray(B)[i],feature_names=names);shap.plots.waterfall(e,max_display=12,show=False);plt.tight_layout();plt.savefig(ROOT/f'figures/ghana/ghana_SHAP_waterfall_case_{i+1}.png',dpi=300,bbox_inches='tight');plt.close()
# Lecturer descriptive
l=pd.read_csv('/mnt/data/Teacher_Trust_Human_AI_dataset.csv');cons={'Teacher Trust':'TT','SHAP Explanation Clarity':'SH','Explanation Usability':'EU','Pedagogical Factors':'PF','Perceived Usefulness':'PU','Perceived Ease of Use':'PE','Institutional Support':'IF','Human-AI Collaboration':'HC','Prediction Quality':'PQ','Fairness and Transparency':'FT'}
def lik(s):return pd.to_numeric(s.astype(str).str.extract(r'([1-5])')[0],errors='coerce')
def al(x):x=x.dropna();k=x.shape[1];return k/(k-1)*(1-x.var(ddof=1).sum()/x.sum(1).var(ddof=1))
rr=[]
for name,pfx in cons.items():
    cs=[c for c in l.columns if re.match(rf'^{pfx}\d+:',c)];x=pd.DataFrame({c:lik(l[c]) for c in cs});s=x.mean(1);rr.append({'Construct':name,'Items':len(cs),'N':s.notna().sum(),'Mean':s.mean(),'SD':s.std(ddof=1),'Minimum':s.min(),'Maximum':s.max(),'Cronbach_alpha':al(x)})
ld=pd.DataFrame(rr);ld.to_csv(ROOT/'results/lecturer/table_4_5_construct_statistics.csv',index=False);l['Record Status'].value_counts().rename_axis('Record Status').reset_index(name='Count').to_csv(ROOT/'results/lecturer/record_status_counts.csv',index=False)
ax=ld.set_index('Construct').Mean.sort_values().plot(kind='barh',figsize=(9,6));ax.set_xlabel('Mean (1–5)');ax.set_title('Lecturer Evaluation Construct Means');plt.tight_layout();plt.savefig(ROOT/'figures/lecturer/lecturer_construct_means.png',dpi=300,bbox_inches='tight');plt.close()
ax=ld.set_index('Construct').Cronbach_alpha.sort_values().plot(kind='barh',figsize=(9,6));ax.axvline(.7,linestyle='--');ax.set_title('Lecturer Construct Reliability');plt.tight_layout();plt.savefig(ROOT/'figures/lecturer/lecturer_construct_reliability.png',dpi=300,bbox_inches='tight');plt.close()
# Code package: create a concise canonical script from fast builder but CV disabled by note
shutil.copy2('/mnt/data/fast_build.py',ROOT/'code/full_rebuild_with_optional_10fold_cv.py');shutil.copy2('/mnt/data/finish_package.py',ROOT/'code/fast_canonical_rebuild.py')
nb=nbf.v4.new_notebook();nb.cells=[nbf.v4.new_markdown_cell('# Updated XGBoost–SHAP Canonical Analysis\nRegenerates benchmark, Ghanaian institutional, SHAP and lecturer outputs.'),nbf.v4.new_code_cell('%run ../code/fast_canonical_rebuild.py')];nbf.write(nb,ROOT/'notebooks/Updated_XGBoost_SHAP_Canonical.ipynb')
(ROOT/'requirements.txt').write_text(Path('/mnt/data/requirements.txt').read_text())
(ROOT/'README.md').write_text('''# Albert XGBoost–SHAP Updated Reproducibility Package\n\nAll requested manuscript figures are saved at 300 dpi.\n\n## Benchmark figures\n- Figure 4.2 model-performance comparison\n- Figure 4.3 ROC curves\n- Precision–Recall curves\n- Four individual confusion matrices + combined matrix panel\n- SHAP global bar, beeswarm, and three local waterfall explanations\n\n## Ghanaian institutional figures\nA separate complete set of model-performance, ROC, PR, confusion-matrix and SHAP figures is in `figures/ghana`. The early-warning model excludes the final exam and total score to avoid target leakage.\n\n## Code\n- `code/fast_canonical_rebuild.py`: fast deterministic rebuild of all requested figures\n- `code/full_rebuild_with_optional_10fold_cv.py`: expanded workflow containing the 10-fold CV section\n- `notebooks/Updated_XGBoost_SHAP_Canonical.ipynb`: notebook entry point\n\nRandom seed = 42; stratified 80:20 held-out test split; SMOTE applied to training data only.\n''')
items=[]
for p in ROOT.rglob('*'):
    if p.is_file():items.append({'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size})
pd.DataFrame(items).sort_values('path').to_csv(ROOT/'MANIFEST.csv',index=False)
print(gd.to_string(index=False));print(ld[['Construct','Mean','SD','Cronbach_alpha']].to_string(index=False));print('files',len(items))
