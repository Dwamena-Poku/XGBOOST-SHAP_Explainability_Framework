"""SCRIPT 7 — categorical/one-hot SHAP audit. CONSUMES Script 1 + Script 5; REFITS NOTHING.
Does not reconcile new ranking to old manuscript tables.
"""
import re, pandas as pd, numpy as np, matplotlib.pyplot as plt, shap
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT)
contract=load_json(D['artifacts']/'script1_contract.json'); rank=pd.read_csv(D['results']/'script5_shap_ranking_encoded.csv')
# Aggregate one-hot levels back to source feature using ColumnTransformer naming convention cat__Course_value.
def source_feature(encoded):
    s=str(encoded)
    if s.startswith('num__'): return s[5:]
    if s.startswith('cat__'):
        body=s[5:]
        # match declared category names longest-first so underscores in names are preserved
        cfg=load_json(D['artifacts']/'script0_schema.json')
        for c in sorted(cfg.get('categorical_columns',[]),key=len,reverse=True):
            if body==c or body.startswith(c+'_'): return c
        return body.split('_',1)[0]
    return s
matrix_path=D['artifacts']/'script5_per_student_shap_matrix.csv'
if not matrix_path.exists():
    raise FileNotFoundError('Run Script 5 to generate the per-student SHAP matrix first.')
matrix=pd.read_csv(matrix_path)
feature_cols=[c for c in matrix.columns if c!='row_index']
source_names=[source_feature(c) for c in feature_cols]
source_order=list(dict.fromkeys(source_names))
source_values=np.column_stack([matrix[[f for f,g in zip(feature_cols,source_names) if g==source]].sum(axis=1).to_numpy() for source in source_order])
agg=pd.DataFrame({'source_feature':source_order,'mean_abs_shap':np.abs(source_values).mean(axis=0)})
agg=agg.sort_values('mean_abs_shap',ascending=False).reset_index(drop=True)
agg['rank']=np.arange(1,len(agg)+1)
agg.to_csv(D['results']/'script7_shap_ranking_source_features.csv',index=False)
pd.DataFrame(source_values,columns=source_order).assign(row_index=matrix['row_index']).to_csv(D['artifacts']/'script7_per_student_source_shap_matrix.csv',index=False)
rank['source_feature']=rank['feature'].map(source_feature)
course_rows=rank[rank['source_feature'].str.lower().eq('course')]
pd.DataFrame([{'check':'Course encoded as one-hot levels','passed':bool(len(course_rows)>1),'n_course_encoded_columns':int(len(course_rows))}]).to_csv(D['results']/'script7_course_encoding_audit.csv',index=False)
ordered=agg['source_feature'].head(20).tolist()[::-1]
plt.figure(figsize=(10,8)); plt.barh(ordered,agg.set_index('source_feature').loc[ordered,'mean_abs_shap']);plt.xlabel('Mean absolute aggregated SHAP (model margin)');plt.tight_layout();plt.savefig(D['figures']/'shap_global_bar_original_variables.png',dpi=220);plt.close()
# For a source-feature beeswarm, plot the signed aggregated attributions; no fabricated source values.
shap.summary_plot(source_values,np.zeros_like(source_values),feature_names=source_order,show=False,max_display=20,color_bar=False)
plt.tight_layout();plt.savefig(D['figures']/'shap_beeswarm_original_variables.png',dpi=220,bbox_inches='tight');plt.close()
print('SCRIPT 7 COMPLETE — per-student signed aggregation before absolute value.')
