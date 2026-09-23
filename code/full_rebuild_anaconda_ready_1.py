# -*- coding: utf-8 -*-
"""
Full rebuild of the XGBoost-SHAP analysis package (Anaconda / Spyder / Jupyter ready).

Put this script and the three CSV files in the SAME folder:
    ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv         (benchmark, UCI dropout data)
    ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv (Ghanaian early-warning model)
    ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv       (lecturer survey, descriptive)
Then run it. Output goes to a new folder "Albert_XGBoost_SHAP_Updated_Package"
next to the script. Missing Python packages are installed automatically.
"""
import sys, subprocess, importlib, warnings

# ----------------------------------------------------------------------------
# 0. Make sure every required package is available (auto-install if missing)
# ----------------------------------------------------------------------------
REQUIRED = [('numpy', 'numpy'), ('pandas', 'pandas'), ('matplotlib', 'matplotlib'),
            ('scikit-learn', 'sklearn'), ('imbalanced-learn', 'imblearn'),
            ('xgboost', 'xgboost'), ('shap', 'shap'), ('nbformat', 'nbformat')]
for _pkg, _mod in REQUIRED:
    try:
        importlib.import_module(_mod)
    except ImportError:
        print(f'Installing missing package: {_pkg} ...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', _pkg])
        except Exception as _e:
            raise SystemExit(f'Could not install {_pkg} ({_e}). In Anaconda Prompt run:  pip install {_pkg}')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                      # save figures to files; no pop-up windows needed
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from xgboost import XGBClassifier
import shap
import nbformat as nbf

SEED = 42

# ----------------------------------------------------------------------------
# 1. Locations and the three input datasets
# ----------------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:                           # e.g. code pasted into a notebook cell
    BASE_DIR = Path.cwd()

UCI_FILE = 'ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv'
GHANA_FILE = 'ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv'
TEACHER_FILE = 'ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv'


def find_file(name):
    """Look for a dataset next to the script, in a data/ sub-folder, or in the working folder."""
    for folder in [BASE_DIR, BASE_DIR / 'data', BASE_DIR.parent / 'data', Path.cwd(), Path.cwd() / 'data']:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Required dataset '{name}' not found. Place it in the same folder as this script: {BASE_DIR}")


def read_csv_any(path):
    """Read a CSV (handles the UTF-8 BOM Excel adds, and ';' separated files)."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding='utf-8-sig', sep=';')
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


benchmark_file = find_file(UCI_FILE)
ghana_file = find_file(GHANA_FILE)
teacher_file = find_file(TEACHER_FILE)
print('Datasets found:\n ', benchmark_file, '\n ', ghana_file, '\n ', teacher_file)

# Output folder (if the script is run from <package>/code, reuse that package folder)
if BASE_DIR.name == 'code' and (BASE_DIR.parent / 'data').is_dir():
    ROOT = BASE_DIR.parent
else:
    ROOT = BASE_DIR / 'Albert_XGBoost_SHAP_Updated_Package'
for d in ['code', 'notebooks', 'figures/benchmark', 'figures/ghana', 'figures/lecturer',
          'results/benchmark', 'results/ghana', 'results/lecturer', 'data']:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
for f in (benchmark_file, ghana_file, teacher_file):          # keep a copy of the data in the package
    try:
        shutil.copy2(f, ROOT / 'data' / f.name)
    except shutil.SameFileError:
        pass

# ----------------------------------------------------------------------------
# 2. Models and helper functions
# ----------------------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(C=1, max_iter=3000, random_state=SEED),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=1, random_state=SEED, n_jobs=1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05, subsample=.9, colsample_bytree=.9,
                             random_state=SEED, eval_metric='logloss', n_jobs=1),
}
warnings_log = []


def metrics(y, p, pr):
    return {'Accuracy': accuracy_score(y, p), 'Precision': precision_score(y, p, zero_division=0),
            'Recall': recall_score(y, p, zero_division=0), 'F1': f1_score(y, p, zero_division=0),
            'ROC_AUC': roc_auc_score(y, pr), 'AUC_PR': average_precision_score(y, pr)}


def plots(df, y, probs, preds, out, prefix, fig42=False):
    ax = df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'AUC_PR']].plot(kind='bar', figsize=(11, 6))
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score'); ax.set_title('Model Performance Comparison')
    plt.xticks(rotation=18, ha='right'); plt.tight_layout()
    fn = 'Figure_4_2_model_performance_comparison.png' if fig42 else f'{prefix}model_performance_comparison.png'
    plt.savefig(out / fn, dpi=300, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(7, 6))
    for n, pr in probs.items():
        fpr, tpr, _ = roc_curve(y, pr)
        plt.plot(fpr, tpr, label=f'{n} (AUC={roc_auc_score(y, pr):.3f})')
    plt.plot([0, 1], [0, 1], '--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves'); plt.legend(fontsize=8); plt.tight_layout()
    rfn = 'Figure_4_3_ROC_curves.png' if fig42 else f'{prefix}ROC_curves.png'
    plt.savefig(out / rfn, dpi=300, bbox_inches='tight'); plt.close()

    plt.figure(figsize=(7, 6)); prev = np.mean(y)
    for n, pr in probs.items():
        pre_, rec_, _ = precision_recall_curve(y, pr)
        plt.plot(rec_, pre_, label=f'{n} (AP={average_precision_score(y, pr):.3f})')
    plt.axhline(prev, linestyle='--', label=f'Prevalence={prev:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall Curves'); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out / f'{prefix}Precision_Recall_curves.png', dpi=300, bbox_inches='tight'); plt.close()

    for n, p in preds.items():
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_predictions(y, p, display_labels=['No intervention', 'Intervention'], ax=ax, colorbar=False)
        ax.set_title(f'{n} Confusion Matrix'); plt.tight_layout()
        safe = re.sub('[^a-z0-9]+', '_', n.lower()).strip('_')
        plt.savefig(out / f'{prefix}confusion_matrix_{safe}.png', dpi=300, bbox_inches='tight'); plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (n, p) in zip(axs.flat, preds.items()):
        ConfusionMatrixDisplay.from_predictions(y, p, display_labels=['No', 'Intervention'], ax=ax, colorbar=False)
        ax.set_title(n)
    plt.tight_layout(); plt.savefig(out / f'{prefix}confusion_matrices_all_models.png', dpi=300, bbox_inches='tight'); plt.close()


def shap_values_for(model, X):
    """SHAP values (n_samples x n_features) and base values for a fitted XGBoost model.
    Falls back to XGBoost's own exact TreeSHAP if shap.TreeExplainer and the installed
    xgboost version are incompatible."""
    try:
        sv = shap.TreeExplainer(model)(X)
        v = np.asarray(sv.values)
        v = v[:, :, 1] if v.ndim == 3 else v
        base = np.asarray(sv.base_values)
        base = base[:, 1] if base.ndim == 2 else base
        base = np.ravel(base)
        if base.size != len(X):
            base = np.full(len(X), base[0])
        return v, base
    except Exception as e:
        print('  (shap.TreeExplainer failed -> using XGBoost native SHAP contributions):', e)
        contrib = model.get_booster().predict(xgb.DMatrix(np.asarray(X)), pred_contribs=True)
        return contrib[:, :-1], contrib[:, -1]


def try_plot(label, func):
    """A failure in one SHAP figure is reported but does not stop the whole run."""
    try:
        func()
    except Exception as e:
        plt.close('all')
        msg = f'WARNING: could not create {label}: {e}'
        print(msg); warnings_log.append(msg)


def shapplots(model, X, features, out, prefix, resdir):
    v, base = shap_values_for(model, X)
    X = np.asarray(X)
    pd.DataFrame({'Feature': features, 'MeanAbsSHAP': np.abs(v).mean(0)}) \
        .sort_values('MeanAbsSHAP', ascending=False).to_csv(resdir / f'{prefix}shap_global_feature_ranking.csv', index=False)

    def bar():
        shap.summary_plot(v, X, feature_names=features, plot_type='bar', show=False, max_display=15)
        plt.tight_layout(); plt.savefig(out / f'{prefix}SHAP_global_bar.png', dpi=300, bbox_inches='tight'); plt.close()

    def swarm():
        shap.summary_plot(v, X, feature_names=features, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(out / f'{prefix}SHAP_beeswarm.png', dpi=300, bbox_inches='tight'); plt.close()

    def waterfall(i):
        def _f():
            e = shap.Explanation(values=v[i], base_values=base[i], data=X[i], feature_names=features)
            shap.plots.waterfall(e, max_display=12, show=False)
            plt.tight_layout(); plt.savefig(out / f'{prefix}SHAP_waterfall_case_{i + 1}.png', dpi=300, bbox_inches='tight'); plt.close()
        return _f

    try_plot(f'{prefix}SHAP_global_bar', bar)
    try_plot(f'{prefix}SHAP_beeswarm', swarm)
    for i in range(min(3, len(X))):
        try_plot(f'{prefix}SHAP_waterfall_case_{i + 1}', waterfall(i))


# ----------------------------------------------------------------------------
# 3. Benchmark: UCI student dropout data
# ----------------------------------------------------------------------------
print('\n[1/3] Benchmark (UCI) ...')
b = read_csv_any(benchmark_file)
if 'Target' not in b.columns:
    raise ValueError("The UCI file must contain a 'Target' column.")
yb = (b['Target'].astype(str).str.strip() == 'Dropout').astype(int)
if yb.nunique() != 2:
    raise ValueError("The Target column must contain both 'Dropout' and non-'Dropout' classes.")
Xb = pd.get_dummies(b.drop(columns='Target'), drop_first=False, dtype=float)   # only encodes text columns
Xb = Xb.apply(pd.to_numeric, errors='coerce').fillna(0)

Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=.2, stratify=yb, random_state=SEED)
if ytr.value_counts().min() < 2:
    raise ValueError('SMOTE needs at least 2 minority-class samples in the training split.')
Xres, yres = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
sc = StandardScaler(); XresS = sc.fit_transform(Xres); XteS = sc.transform(Xte)

rows = []; probs = {}; preds = {}; fitted = {}
for n, m in models.items():
    m.fit(XresS, yres)
    pr = m.predict_proba(XteS)[:, 1]; p = (pr >= .5).astype(int)
    rows.append({'Model': n, **metrics(yte, p, pr)}); probs[n] = pr; preds[n] = p; fitted[n] = m
bd = pd.DataFrame(rows)
bd.to_csv(ROOT / 'results/benchmark/heldout_test_metrics.csv', index=False)
plots(bd, yte, probs, preds, ROOT / 'figures/benchmark', 'benchmark_', True)
shapplots(fitted['XGBoost'], XteS, list(Xb.columns), ROOT / 'figures/benchmark', 'benchmark_', ROOT / 'results/benchmark')

print('  10-fold cross-validation (takes a few minutes) ...')
cv = StratifiedKFold(10, shuffle=True, random_state=SEED); cvrows = []
scoring = {'ROC_AUC': 'roc_auc', 'AUC_PR': 'average_precision', 'F1': 'f1'}
for n, m in models.items():
    pipe = ImbPipeline([('scale', StandardScaler()), ('smote', SMOTE(random_state=SEED)), ('model', m)])
    res = cross_validate(pipe, Xb, yb, cv=cv, scoring=scoring, n_jobs=1)
    for metric in scoring:
        s = res[f'test_{metric}']
        cvrows.append({'Model': n, 'Metric': metric, 'Mean': s.mean(), 'SD': s.std(ddof=1)})
pd.DataFrame(cvrows).to_csv(ROOT / 'results/benchmark/cross_validation_summary.csv', index=False)

# ----------------------------------------------------------------------------
# 4. Ghanaian institutional early-warning model
# ----------------------------------------------------------------------------
print('\n[2/3] Ghanaian institutional data ...')
gh = read_csv_any(ghana_file)
required_g = ['Course', 'CA_20', 'Intervention']
missing = [c for c in required_g + ['Overall_Score'] if c not in gh.columns]
if missing:
    raise ValueError(f'Ghana dataset is missing columns: {missing}')
gc = gh.dropna(subset=required_g).copy()
gc['Intervention'] = pd.to_numeric(gc['Intervention'], errors='coerce')
gc = gc.dropna(subset=['Intervention'])
Xg = gc[['CA_20', 'Course']]; yg = gc['Intervention'].astype(int)

try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:                          # older scikit-learn (<1.2)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
pre = ColumnTransformer([('num', StandardScaler(), ['CA_20']), ('cat', ohe, ['Course'])])
Xt, Xv, yt, yv = train_test_split(Xg, yg, test_size=.2, stratify=yg, random_state=SEED)
A = pre.fit_transform(Xt); B = pre.transform(Xv)
Ar, yr = SMOTE(random_state=SEED).fit_resample(A, yt)

rows = []; probs = {}; preds = {}; gf = {}
for n, m in models.items():
    mm = clone(m); mm.fit(Ar, yr)
    pr = mm.predict_proba(B)[:, 1]; p = (pr >= .5).astype(int)
    rows.append({'Model': n, **metrics(yv, p, pr)}); probs[n] = pr; preds[n] = p; gf[n] = mm
gd = pd.DataFrame(rows)
gd.to_csv(ROOT / 'results/ghana/heldout_test_metrics.csv', index=False)
plots(gd, yv, probs, preds, ROOT / 'figures/ghana', 'ghana_', False)
shapplots(gf['XGBoost'], B, list(pre.get_feature_names_out()), ROOT / 'figures/ghana', 'ghana_', ROOT / 'results/ghana')
overall = pd.to_numeric(gc['Overall_Score'], errors='coerce')
pd.DataFrame({'Measure': ['Initial records', 'Complete cases', 'Intervention cases', 'Non-intervention cases',
                          'Intervention prevalence', 'Overall mean', 'Overall SD'],
              'Value': [len(gh), len(gc), int(yg.sum()), int((1 - yg).sum()), yg.mean(), overall.mean(), overall.std(ddof=1)]}
             ).to_csv(ROOT / 'results/ghana/dataset_summary.csv', index=False)

# ----------------------------------------------------------------------------
# 5. Lecturer (teacher trust) survey - descriptive statistics
# ----------------------------------------------------------------------------
print('\n[3/3] Lecturer survey ...')
l = read_csv_any(teacher_file)
cons = {'Teacher Trust': 'TT', 'SHAP Explanation Clarity': 'SH', 'Explanation Usability': 'EU',
        'Pedagogical Factors': 'PF', 'Perceived Usefulness': 'PU', 'Perceived Ease of Use': 'PE',
        'Institutional Support': 'IF', 'Human-AI Collaboration': 'HC', 'Prediction Quality': 'PQ',
        'Fairness and Transparency': 'FT'}


def lik(s):
    return pd.to_numeric(s.astype(str).str.extract(r'([1-5])')[0], errors='coerce')


def alpha(x):
    x = x.dropna(); k = x.shape[1]
    return k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))


rr = []
for name, pfx in cons.items():
    cs = [c for c in l.columns if re.match(rf'^{pfx}\d+:', c)]
    if not cs:
        raise ValueError(f'No survey items found for construct {name} ({pfx}1:, {pfx}2: ...)')
    x = pd.DataFrame({c: lik(l[c]) for c in cs}); s = x.mean(axis=1)
    rr.append({'Construct': name, 'Items': len(cs), 'N': int(s.notna().sum()), 'Mean': s.mean(), 'SD': s.std(ddof=1),
               'Minimum': s.min(), 'Maximum': s.max(), 'Cronbach_alpha': alpha(x)})
ld = pd.DataFrame(rr)
ld.to_csv(ROOT / 'results/lecturer/table_4_5_construct_statistics.csv', index=False)
if 'Record Status' in l.columns:
    l['Record Status'].value_counts().rename_axis('Record Status').reset_index(name='Count') \
        .to_csv(ROOT / 'results/lecturer/record_status_counts.csv', index=False)
ax = ld.set_index('Construct').Mean.sort_values().plot(kind='barh', figsize=(9, 6))
ax.set_xlabel('Mean (1–5)'); plt.tight_layout()
plt.savefig(ROOT / 'figures/lecturer/lecturer_construct_means.png', dpi=300, bbox_inches='tight'); plt.close()

# ----------------------------------------------------------------------------
# 6. Package files: script copy, notebook, requirements, README, manifest
# ----------------------------------------------------------------------------
try:
    me = Path(__file__).resolve()
    dest = ROOT / 'code' / 'rebuild_all_analysis.py'
    if me != dest.resolve():
        shutil.copy2(me, dest)
except NameError:
    print('NOTE: script location unknown (not run from a file); code copy omitted from package.')

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell('# Canonical XGBoost–SHAP Analysis'),
            nbf.v4.new_code_cell('%run ../code/rebuild_all_analysis.py')]
nbf.write(nb, ROOT / 'notebooks/Updated_XGBoost_SHAP_Analysis.ipynb')

(ROOT / 'requirements.txt').write_text(
    'numpy\npandas\nmatplotlib\nscikit-learn>=1.2\nimbalanced-learn\nxgboost\nshap\nnbformat\n', encoding='utf-8')

(ROOT / 'README.md').write_text('''# Updated XGBoost–SHAP Reproducibility Package

All requested figures are regenerated from a single canonical run. Figures are saved as 300-dpi PNG files.

## Input data (in `data/`)
- `ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv` - benchmark UCI dropout dataset
- `ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv` - Ghanaian institutional dataset
- `ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv` - lecturer survey

## Manuscript figures
- Figure 4.2: `figures/benchmark/Figure_4_2_model_performance_comparison.png`
- Figure 4.3: `figures/benchmark/Figure_4_3_ROC_curves.png`
- Precision–Recall curves, individual and combined confusion matrices, and SHAP global/beeswarm/waterfall figures are in `figures/benchmark/`.
- Ghanaian institutional equivalents are in `figures/ghana/`.

## Canonical settings
Random seed 42; stratified 80:20 split; SMOTE applied only to training data; fixed tuned model parameters; benchmark 10-fold CV summary included.

## Ghanaian early-warning model
Intervention is defined as Overall_Score < 50. Exam_60 and Overall_Score are excluded from inputs to prevent target leakage.

## Lecturer data
The supplied 350-row file is analysed separately. Check `results/lecturer/record_status_counts.csv` before making collection-status claims.
''', encoding='utf-8')

items = [{'path': str(p.relative_to(ROOT)), 'size_bytes': p.stat().st_size} for p in ROOT.rglob('*') if p.is_file()]
pd.DataFrame(items).sort_values('path').to_csv(ROOT / 'MANIFEST.csv', index=False)

print('\nBENCH\n', bd.to_string(index=False))
print('\nGHANA\n', gd.to_string(index=False))
print('\nLECTURER\n', ld.to_string(index=False))
print('\nfiles', len(items))
if warnings_log:
    print('\nCompleted with warnings:'); [print(' -', w) for w in warnings_log]
print('\nDONE. Output folder:', ROOT)
