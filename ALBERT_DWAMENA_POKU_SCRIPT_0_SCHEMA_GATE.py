"""SCRIPT 0 — dataset/schema bootstrap and integrity gate.
ANCHOR: REM-ALBERT_DWAMENA_POKU-2026-08-26
ALBERT DWAMENA-POKU ONLY.
"""
from ALBERT_DWAMENA_POKU_COMMON import *
ROOT=root_from_script(__file__); D=dirs(ROOT)
student_path=r"C:\Users\GODLY VIEW COMPUTERS\Documents\THESIS 2026\XGBOOST-SHAP_Explainability_Framework-main\data\ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv"
teacher_path=r"C:\Users\GODLY VIEW COMPUTERS\Documents\THESIS 2026\XGBOOST-SHAP_Explainability_Framework-main\data\ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv"
institutional_path=r"C:\Users\GODLY VIEW COMPUTERS\Documents\THESIS 2026\XGBOOST-SHAP_Explainability_Framework-main\data\ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv"
for p in [student_path, teacher_path, institutional_path]:
    if not p.exists(): raise FileNotFoundError(f'Missing required dataset: {p}')
df=load_table(student_path)
# Real headers harvested from the included dataset; no guessed aliases.
TARGET='Target'
CAT_COLS=['Marital status','Application mode','Course','Daytime/evening attendance','Previous qualification','Nacionality',
'Mother\'s qualification','Father\'s qualification','Mother\'s occupation','Father\'s occupation','Displaced','Educational special needs',
'Debtor','Tuition fees up to date','Gender','Scholarship holder','International']
NUM_COLS=[c for c in df.columns if c not in CAT_COLS+[TARGET]]
missing=[c for c in CAT_COLS+NUM_COLS+[TARGET] if c not in df.columns]
if missing: raise KeyError(f'Dataset/schema mismatch: {missing}')
cfg={
 'student_dataset_path':'data/ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv',
 'teacher_dataset_path':'data/ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv',
 'institutional_dataset_path':'data/ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv',
 'target_column':TARGET, 'positive_class':'Dropout', 'target_mapping':{'Dropout':1,'Graduate':0,'Enrolled':0},
 'id_columns':[], 'categorical_columns':CAT_COLS, 'numeric_columns':NUM_COLS,
 'test_size':0.20,'random_state':42,'cv_folds':10
}
save_json(cfg,D['artifacts']/'script0_schema.json')
status={
 'anchor':ANCHOR,'student_rows':int(len(df)),'student_columns':int(df.shape[1]),
 'teacher_file_loads':True,'teacher_rows':int(len(load_table(teacher_path))),
 'institutional_file_loads':True,'institutional_rows':int(len(load_table(institutional_path))),
 'student_sha256':sha256(student_path),'teacher_sha256':sha256(teacher_path),'institutional_sha256':sha256(institutional_path)
}
save_json(status,D['artifacts']/'script0_data_gate.json')
print('SCRIPT 0 COMPLETE')
print(json.dumps(status,indent=2))
