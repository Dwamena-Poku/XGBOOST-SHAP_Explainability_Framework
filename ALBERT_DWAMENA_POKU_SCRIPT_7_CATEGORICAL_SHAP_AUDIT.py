"""SCRIPT 7 — categorical/one-hot SHAP audit. CONSUMES Script 1 + Script 5; REFITS NOTHING.
Does not reconcile new ranking to old manuscript tables.
"""
import re, pandas as pd
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
rank['source_feature']=rank['feature'].map(source_feature)
agg=rank.groupby('source_feature',as_index=False)['mean_abs_shap'].sum().sort_values('mean_abs_shap',ascending=False)
agg['rank']=range(1,len(agg)+1); agg.to_csv(D['results']/'script7_shap_ranking_source_features.csv',index=False)
# mechanical Course audit
course_rows=rank[rank['source_feature'].str.lower().eq('course')]
pd.DataFrame([{'check':'Course encoded as one-hot levels','passed':bool(len(course_rows)>1),'n_course_encoded_columns':int(len(course_rows)),'note':'Ranking is reported as computed; never reconciled to the old table.'}]).to_csv(D['results']/'script7_course_encoding_audit.csv',index=False)
print('SCRIPT 7 COMPLETE — no refit performed. Do not force this ranking to match the manuscript.')
