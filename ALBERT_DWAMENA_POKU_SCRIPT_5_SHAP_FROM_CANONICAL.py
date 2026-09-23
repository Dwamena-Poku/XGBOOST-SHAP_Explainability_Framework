"""SCRIPT 5 — SHAP from Script 1 canonical XGBoost. CONSUMES artifacts; REFITS NOTHING."""
import joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt, shap
from pathlib import Path
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT); cfg=load_json(D['artifacts']/'script0_schema.json'); require_no_fillins(cfg)
p=resolve_data_path(ROOT,cfg['student_dataset_path']); df=load_table(p); X=df[cfg.get('numeric_columns',[])+cfg.get('categorical_columns',[])]
contract=load_json(D['artifacts']/'script1_contract.json'); pipe=joblib.load(D['artifacts']/'script1_model_xgboost.joblib')
splits=pd.read_csv(D['artifacts']/'script1_split_indices.csv'); te=splits['test_index'].dropna().astype(int).to_numpy(); Xte=X.iloc[te]
Xt=pipe.named_steps['pre'].transform(Xte); names=contract['feature_names']; model=pipe.named_steps['model']
expl=shap.TreeExplainer(model); sv=expl(Xt)
meanabs=np.abs(sv.values).mean(axis=0); rank=pd.DataFrame({'feature':names,'mean_abs_shap':meanabs}).sort_values('mean_abs_shap',ascending=False); rank.to_csv(D['results']/'script5_shap_ranking_encoded.csv',index=False)
shap.summary_plot(sv.values,Xt,feature_names=names,show=False); plt.tight_layout(); plt.savefig(D['figures']/'shap_beeswarm.png',dpi=300,bbox_inches='tight'); plt.close()
shap.summary_plot(sv.values,Xt,feature_names=names,plot_type='bar',show=False); plt.tight_layout(); plt.savefig(D['figures']/'shap_global_bar.png',dpi=300,bbox_inches='tight'); plt.close()
for i in range(min(3,len(Xte))):
    shap.plots.waterfall(sv[i],show=False,max_display=12); plt.tight_layout(); plt.savefig(D['figures']/f'shap_waterfall_case_{i+1}.png',dpi=300,bbox_inches='tight'); plt.close()
print('SCRIPT 5 COMPLETE — no refit performed.')
