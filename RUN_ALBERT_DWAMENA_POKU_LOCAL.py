from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parent
order=[
'ALBERT_DWAMENA_POKU_SCRIPT_0_SCHEMA_GATE.py',
'ALBERT_DWAMENA_POKU_SCRIPT_1_CANONICAL_FIT.py',
'ALBERT_DWAMENA_POKU_SCRIPT_2_HELDOUT_FIGURES.py',
'ALBERT_DWAMENA_POKU_SCRIPT_3_EQUAL_TUNING_AUDIT.py',
'ALBERT_DWAMENA_POKU_SCRIPT_4_MULTI_SEED_STABILITY.py',
'ALBERT_DWAMENA_POKU_SCRIPT_5_SHAP_FROM_CANONICAL.py',
'ALBERT_DWAMENA_POKU_SCRIPT_6_PHASE3_GATED_ABLATION.py',
'ALBERT_DWAMENA_POKU_SCRIPT_7_CATEGORICAL_SHAP_AUDIT.py',
'ALBERT_DWAMENA_POKU_SCRIPT_8_DEPENDENCE_CORRECTED_TESTS.py']
for s in order:
 print('\n===',s,'===')
 r=subprocess.run([sys.executable,str(ROOT/s)],cwd=ROOT)
 if r.returncode!=0: raise SystemExit(f'{s} failed with code {r.returncode}')
print('\nCHAIN COMPLETE. Script 6 may report NOT RUN until approved Phase 3 parameters are supplied.')
