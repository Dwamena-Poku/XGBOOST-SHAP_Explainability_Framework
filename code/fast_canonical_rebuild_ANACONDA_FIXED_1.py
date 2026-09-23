"""
Fast Canonical XGBoost-SHAP Rebuild — Anaconda-safe version
------------------------------------------------------------
Put this script in the root of your thesis project, next to a
"data" folder (or just next to the CSVs directly).

Expected data files (place in <project>/data/ or next to the script):
    * ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv   (course-level GPA/CA records)
    * ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv          (lecturer trust survey)
    * ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv            (UCI student dropout dataset)

The script creates all output folders automatically:
    results/ghana, results/uci, results/lecturer
    figures/ghana, figures/uci, figures/lecturer
"""

from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SEED = 42

RESULTS_GHANA = ROOT / "results" / "ghana"
RESULTS_UCI = ROOT / "results" / "uci"
RESULTS_LECTURER = ROOT / "results" / "lecturer"
FIGURES_GHANA = ROOT / "figures" / "ghana"
FIGURES_UCI = ROOT / "figures" / "uci"
FIGURES_LECTURER = ROOT / "figures" / "lecturer"

for folder in [
    RESULTS_GHANA, RESULTS_UCI, RESULTS_LECTURER,
    FIGURES_GHANA, FIGURES_UCI, FIGURES_LECTURER
]:
    folder.mkdir(parents=True, exist_ok=True)

# Exact filenames of the three datasets this script uses
GHANA_FILENAME = "ALBERT_DWAMENA_POKU_GHANA_INSTITUTIONAL_DATA.csv"
TEACHER_FILENAME = "ALBERT_DWAMENA_POKU_TEACHER_TRUST_DATA.csv"
UCI_FILENAME = "ALBERT_DWAMENA_POKU_UCI_STUDENT_DATA.csv"


def find_file(filename):
    """Find a named file anywhere under the project folder."""
    matches = list(ROOT.rglob(filename))
    if matches:
        return matches[0]
    return None


def make_onehot():
    """Support both old and new scikit-learn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def metrics(y, pred, prob):
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, prob),
        "AUC_PR": average_precision_score(y, prob),
    }


def plotset(df, y, probs, preds, out, prefix, title_stub="Model"):
    """Create model comparison, ROC, PR and confusion-matrix figures."""

    ax = df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "AUC_PR"]
    ].plot(kind="bar", figsize=(11, 6))

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{title_stub} Model Performance")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(
        out / f"{prefix}model_performance_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={roc_auc_score(y, prob):.3f})"
        )

    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_stub} ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        out / f"{prefix}ROC_curves.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(7, 6))
    prevalence = np.mean(y)

    for name, prob in probs.items():
        precision, recall, _ = precision_recall_curve(y, prob)
        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={average_precision_score(y, prob):.3f})"
        )

    plt.axhline(
        prevalence,
        linestyle="--",
        label=f"Prevalence={prevalence:.3f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_stub} Precision–Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        out / f"{prefix}Precision_Recall_curves.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    for name, pred in preds.items():
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y,
            pred,
            display_labels=["No intervention", "Intervention"],
            ax=ax,
            colorbar=False
        )
        ax.set_title(f"{name} Confusion Matrix")
        plt.tight_layout()

        safe = re.sub(
            r"[^a-z0-9]+",
            "_",
            name.lower()
        ).strip("_")

        plt.savefig(
            out / f"{prefix}confusion_matrix_{safe}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (name, pred) in zip(axes.flat, preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y,
            pred,
            display_labels=["No", "Intervention"],
            ax=ax,
            colorbar=False
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(
        out / f"{prefix}confusion_matrices_all_models.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def fresh_models():
    """A brand-new, unfitted copy of each model."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1, max_iter=3000, random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=3, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=8, random_state=SEED, n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED,
            eval_metric="logloss", n_jobs=1
        ),
    }


def run_classification_pipeline(
    X, y, preprocessor, results_dir, figures_dir, prefix, title_stub
):
    """Train/test split -> SMOTE -> fit 4 models -> metrics -> plots.

    Returns (metrics_df, fitted_models, fitted_preprocessor,
             transformed_holdout_X, holdout_y).
    """
    Xt, Xv, yt, yv = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    A = preprocessor.fit_transform(Xt)
    B = preprocessor.transform(Xv)

    Ar, yr = SMOTE(random_state=SEED).fit_resample(A, yt)

    rows = []
    probs = {}
    preds = {}
    fits = {}

    for name, model in fresh_models().items():
        fitted = clone(model)
        fitted.fit(Ar, yr)

        probability = fitted.predict_proba(B)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        rows.append({"Model": name, **metrics(yv, prediction, probability)})

        probs[name] = probability
        preds[name] = prediction
        fits[name] = fitted

    result_df = pd.DataFrame(rows)
    result_df.to_csv(
        results_dir / f"{prefix}heldout_test_metrics.csv", index=False
    )

    plotset(result_df, yv, probs, preds, figures_dir, prefix, title_stub)

    return result_df, fits, preprocessor, B, yv


def run_shap_analysis(
    xgb_model, preprocessor, B, results_dir, figures_dir, prefix, title_stub
):
    """Global + local SHAP explanations for the fitted XGBoost model."""
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(B)

    if hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
    else:
        values = np.asarray(shap_values)

    if values.ndim == 3:
        values = values[:, :, 1]

    feature_names = list(preprocessor.get_feature_names_out())

    pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": np.abs(values).mean(axis=0)
    }).sort_values(
        "MeanAbsSHAP", ascending=False
    ).to_csv(
        results_dir / f"{prefix}shap_global_feature_ranking.csv",
        index=False
    )

    shap.summary_plot(
        values, B, feature_names=feature_names,
        plot_type="bar", show=False, max_display=15
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{prefix}SHAP_global_bar.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    shap.summary_plot(
        values, B, feature_names=feature_names,
        show=False, max_display=15
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{prefix}SHAP_beeswarm.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    base_values = np.asarray(explainer.expected_value).reshape(-1)

    for i in range(min(3, len(B))):
        base = base_values[0] if len(base_values) == 1 else base_values[1]

        explanation = shap.Explanation(
            values=values[i],
            base_values=base,
            data=np.asarray(B)[i],
            feature_names=feature_names
        )

        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"{prefix}SHAP_waterfall_case_{i + 1}.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close()


# ---------------------------------------------------------------------
# REQUIRED DATA CHECK
# ---------------------------------------------------------------------
ghana_file = find_file(GHANA_FILENAME)
teacher_file = find_file(TEACHER_FILENAME)
uci_file = find_file(UCI_FILENAME)

missing = []
if ghana_file is None:
    missing.append(GHANA_FILENAME)
if teacher_file is None:
    missing.append(TEACHER_FILENAME)
if uci_file is None:
    missing.append(UCI_FILENAME)

if missing:
    print("\nDATA FILES REQUIRED BEFORE RUNNING THE ANALYSIS\n")
    for item in missing:
        print("  -", item)

    print("\nExpected project structure:")
    print(ROOT)
    print("├── fast_canonical_rebuild_ANACONDA_FIXED.py")
    print("├── data")
    print(f"│   ├── {GHANA_FILENAME}")
    print(f"│   ├── {TEACHER_FILENAME}")
    print(f"│   └── {UCI_FILENAME}")
    print("├── results")
    print("└── figures")

    raise FileNotFoundError(
        "\nThe analysis was stopped because required input datasets "
        "were not found. Put the three CSVs above in the project "
        "folder (or its 'data' subfolder) and run the script again."
    )


# ---------------------------------------------------------------------
# GHANA INSTITUTIONAL DATA (course-level continuous assessment records)
# ---------------------------------------------------------------------
gh = pd.read_csv(ghana_file)

required_columns = ["Course", "CA_10_A", "CA_10_B", "CA_20", "Exam_60"]
missing_columns = [c for c in required_columns if c not in gh.columns]

if missing_columns:
    raise KeyError(
        "The Ghana institutional dataset is missing these required "
        "columns: " + str(missing_columns)
    )

for col in ["CA_10_A", "CA_10_B", "CA_20", "Exam_60"]:
    gh[col] = pd.to_numeric(gh[col], errors="coerce")

# -1 is used in the raw export as a "no data" sentinel — treat as missing
gh.loc[gh["CA_20"] < 0, "CA_20"] = np.nan

# Use the pre-computed Overall_Score/Intervention columns when present and
# valid; otherwise (or where missing) recompute from the components.
if "Overall_Score" in gh.columns:
    gh["OverallScore"] = pd.to_numeric(gh["Overall_Score"], errors="coerce")
else:
    gh["OverallScore"] = np.nan

recompute = gh["OverallScore"].isna()
gh.loc[recompute, "OverallScore"] = gh.loc[
    recompute, ["CA_10_A", "CA_10_B", "CA_20", "Exam_60"]
].sum(axis=1, min_count=4)

gc = gh.dropna(subset=["OverallScore", "CA_20"]).copy()

if "Intervention" in gc.columns:
    gc["Intervention"] = pd.to_numeric(
        gc["Intervention"], errors="coerce"
    ).fillna((gc["OverallScore"] < 50).astype(int)).astype(int)
else:
    gc["Intervention"] = (gc["OverallScore"] < 50).astype(int)

gc["CourseCode"] = gc["Course"]

if gc["Intervention"].nunique() < 2:
    raise ValueError(
        "The Ghana intervention target contains only one class. "
        "A binary classification model requires both classes."
    )


# ---------------------------------------------------------------------
# GHANA PREPROCESSING + MODELING
# ---------------------------------------------------------------------
X_gh = gc[["CA_20", "CourseCode"]]
y_gh = gc["Intervention"]

pre_gh = ColumnTransformer([
    ("num", StandardScaler(), ["CA_20"]),
    ("cat", make_onehot(), ["CourseCode"])
])

gd, fits_gh, pre_gh, B_gh, yv_gh = run_classification_pipeline(
    X_gh, y_gh, pre_gh,
    RESULTS_GHANA, FIGURES_GHANA, "ghana_", "Ghanaian Institutional"
)

pd.DataFrame({
    "Measure": [
        "Initial records",
        "Complete cases",
        "Intervention cases",
        "Non-intervention cases",
        "Intervention prevalence",
        "Overall mean",
        "Overall SD"
    ],
    "Value": [
        len(gh),
        len(gc),
        int(y_gh.sum()),
        int((1 - y_gh).sum()),
        y_gh.mean(),
        gc["OverallScore"].mean(),
        gc["OverallScore"].std(ddof=1)
    ]
}).to_csv(RESULTS_GHANA / "dataset_summary.csv", index=False)

run_shap_analysis(
    fits_gh["XGBoost"], pre_gh, B_gh,
    RESULTS_GHANA, FIGURES_GHANA, "ghana_", "Ghanaian"
)


# ---------------------------------------------------------------------
# UCI STUDENT DATA (public dropout/academic-success benchmark dataset)
# ---------------------------------------------------------------------
uci = pd.read_csv(uci_file)

if "Target" not in uci.columns:
    raise KeyError(
        "The UCI student dataset is missing the required 'Target' column."
    )

uci = uci.dropna().copy()

# Binarize to match the same "needs intervention" framing used for Ghana:
# Dropout -> 1 (at risk), Enrolled/Graduate -> 0
uci["Intervention"] = (uci["Target"].str.strip() == "Dropout").astype(int)

feature_cols_uci = [c for c in uci.columns if c not in ("Target", "Intervention")]
for col in feature_cols_uci:
    uci[col] = pd.to_numeric(uci[col], errors="coerce")

uci = uci.dropna(subset=feature_cols_uci).copy()

if uci["Intervention"].nunique() < 2:
    raise ValueError(
        "The UCI intervention target contains only one class. "
        "A binary classification model requires both classes."
    )

X_uci = uci[feature_cols_uci]
y_uci = uci["Intervention"]

pre_uci = ColumnTransformer([
    ("num", StandardScaler(), feature_cols_uci)
])

ud, fits_uci, pre_uci, B_uci, yv_uci = run_classification_pipeline(
    X_uci, y_uci, pre_uci,
    RESULTS_UCI, FIGURES_UCI, "uci_", "UCI Student Dropout"
)

pd.DataFrame({
    "Measure": [
        "Initial records",
        "Complete cases",
        "Dropout cases",
        "Non-dropout cases",
        "Dropout prevalence"
    ],
    "Value": [
        len(uci),
        len(uci),
        int(y_uci.sum()),
        int((1 - y_uci).sum()),
        y_uci.mean()
    ]
}).to_csv(RESULTS_UCI / "dataset_summary.csv", index=False)

run_shap_analysis(
    fits_uci["XGBoost"], pre_uci, B_uci,
    RESULTS_UCI, FIGURES_UCI, "uci_", "UCI Student"
)


# ---------------------------------------------------------------------
# LECTURER DESCRIPTIVE ANALYSIS (teacher trust survey)
# ---------------------------------------------------------------------
lecturer = pd.read_csv(teacher_file)

constructs = {
    "Teacher Trust": "TT",
    "SHAP Explanation Clarity": "SH",
    "Explanation Usability": "EU",
    "Pedagogical Factors": "PF",
    "Perceived Usefulness": "PU",
    "Perceived Ease of Use": "PE",
    "Institutional Support": "IF",
    "Human-AI Collaboration": "HC",
    "Prediction Quality": "PQ",
    "Fairness and Transparency": "FT"
}


def likert(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r"([1-5])")[0],
        errors="coerce"
    )


def cronbach_alpha(data):
    data = data.dropna()
    k = data.shape[1]

    if k < 2:
        return np.nan

    total_variance = data.sum(axis=1).var(ddof=1)

    if total_variance == 0:
        return np.nan

    return (
        k / (k - 1)
        * (1 - data.var(ddof=1).sum() / total_variance)
    )


rows = []

for name, prefix in constructs.items():
    columns = [
        col for col in lecturer.columns
        if re.match(rf"^{re.escape(prefix)}\d+:", col)
    ]

    if not columns:
        continue

    data = pd.DataFrame({col: likert(lecturer[col]) for col in columns})
    score = data.mean(axis=1)

    rows.append({
        "Construct": name,
        "Items": len(columns),
        "N": score.notna().sum(),
        "Mean": score.mean(),
        "SD": score.std(ddof=1),
        "Minimum": score.min(),
        "Maximum": score.max(),
        "Cronbach_alpha": cronbach_alpha(data)
    })

ld = pd.DataFrame(rows)
ld.to_csv(RESULTS_LECTURER / "table_4_5_construct_statistics.csv", index=False)

if "Record Status" in lecturer.columns:
    lecturer["Record Status"].value_counts().rename_axis(
        "Record Status"
    ).reset_index(name="Count").to_csv(
        RESULTS_LECTURER / "record_status_counts.csv", index=False
    )

if not ld.empty:
    ax = ld.set_index("Construct").Mean.sort_values().plot(
        kind="barh", figsize=(9, 6)
    )
    ax.set_xlabel("Mean (1–5)")
    ax.set_title("Lecturer Evaluation Construct Means")
    plt.tight_layout()
    plt.savefig(
        FIGURES_LECTURER / "lecturer_construct_means.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    ax = ld.set_index("Construct").Cronbach_alpha.sort_values().plot(
        kind="barh", figsize=(9, 6)
    )
    ax.axvline(0.7, linestyle="--")
    ax.set_title("Lecturer Construct Reliability")
    plt.tight_layout()
    plt.savefig(
        FIGURES_LECTURER / "lecturer_construct_reliability.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------
items = []
for path in ROOT.rglob("*"):
    if path.is_file():
        items.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size
        })

pd.DataFrame(items).sort_values("path").to_csv(
    ROOT / "MANIFEST.csv", index=False
)


# ---------------------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANACONDA XGBOOST-SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nProject folder:")
print(ROOT)

print("\nGhana model results:")
print(gd.to_string(index=False))

print("\nUCI student model results:")
print(ud.to_string(index=False))

if not ld.empty:
    print("\nLecturer construct results:")
    print(ld[["Construct", "Mean", "SD", "Cronbach_alpha"]].to_string(index=False))

print("\nOutput folders:")
print("  Results :", ROOT / "results")
print("  Figures :", ROOT / "figures")
print("  Manifest:", ROOT / "MANIFEST.csv")
print("\nAll requested analysis steps completed.")
