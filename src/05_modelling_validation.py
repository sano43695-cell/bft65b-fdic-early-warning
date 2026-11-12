"""
src/05_modelling_validation.py – Part 5: Modelling, Validation & Cost-Based Thresholding
========================================================================================

Implements:
  1. Expanding-Window Time-Series Cross-Validation (with a strict 4-quarter purging/embargo
     gap between Train and Validation in every fold) inside the Development period (1984Q1–2008Q4).
  2. Model Suite (all with fixed random_state=42):
       • Simple Baseline : LogisticRegression (class_weight='balanced') on 6 core ratios
       • Full LogReg     : LogisticRegression (class_weight='balanced') on all 37 CAMELS features
       • Random Forest   : RandomForestClassifier on all 37 CAMELS features
       • HistGBDT        : HistGradientBoostingClassifier on all 37 CAMELS features
       • LightGBM        : LGBMClassifier on all 37 CAMELS features
       • Spark MLlib Demo: pyspark.ml Pipeline (Imputer -> VectorAssembler -> StandardScaler
                           -> LogisticRegression) fitted strictly on Train (1984Q1–2006Q4).
  3. Metrics for Time-Ordered, Highly Imbalanced Data:
       • PR-AUC (Average Precision), ROC-AUC, Brier Score
       • Quarterly Supervisory Recall @ Top 1% and Recall @ Top 5% highest-risk banks
  4. Data-Driven Cost Threshold Optimization (on Validation set only):
       • Estimates C_FN (missed failure cost) and C_FP (false-alarm inspection cost)
         strictly from Development-era failures (1984–2008) in failures.parquet.
       • Finds optimal probability threshold p* minimizing Total Cost = FN * C_FN + FP * C_FP
         on the Validation set (2008Q1–2008Q4).
  5. Single Hold-Out Test Evaluation (2010Q1–2025Q2):
       • Evaluates the locked best model and locked threshold p* ONCE on the Test set.

Outputs:
  • outputs/tables/part5_model_comparison.csv
  • outputs/tables/part5_cv_results.csv
  • outputs/figures/part5_pr_roc_curves.png
  • outputs/figures/part5_cost_threshold.png
  • outputs/figures/part5_feature_importance.png
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data_raw" / "fdic"
DATA_PROC  = ROOT / "data_processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS   = ROOT / "outputs" / "figures"

FEATURE_PANEL_PATH = DATA_PROC  / "features_panel.parquet"
FAILURES_PATH      = DATA_RAW   / "failures.parquet"
MODEL_COMP_CSV     = OUT_TABLES / "part5_model_comparison.csv"
CV_RESULTS_CSV     = OUT_TABLES / "part5_cv_results.csv"
COST_META_JSON     = DATA_PROC  / "part5_cost_and_preds.joblib"

FIG_PR_ROC   = OUT_FIGS / "part5_pr_roc_curves.png"
FIG_COST     = OUT_FIGS / "part5_cost_threshold.png"
FIG_FEAT_IMP = OUT_FIGS / "part5_feature_importance.png"

# Baseline 6 core ratios (1 per CAMELS dimension)
BASELINE_FEATURES = [
    "eq_to_asset",          # C: Equity / Assets
    "npl_to_loans",         # A: Noncurrent Loans / Gross Loans
    "cost_to_income",       # M: Noninterest Expense / Operating Revenue
    "roa",                  # E: Return on Assets
    "liquid_assets_ratio",  # L: (Cash + Securities) / Assets
    "log_asset",            # S: Log Total Assets (size)
]


# ════════════════════════════════════════════════════════════════════════════
# 1. LEAK-FREE PREPROCESSING TRANSFORMER (FIT ON TRAIN ONLY)
# ════════════════════════════════════════════════════════════════════════════
class InfAndOutlierClipper(BaseEstimator, TransformerMixin):
    """Replaces ±inf with NaN and clips features to quantiles learned ONLY on X_train."""

    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: np.ndarray | pd.DataFrame, y=None) -> "InfAndOutlierClipper":
        arr = np.asarray(X, dtype=np.float64).copy()
        arr[~np.isfinite(arr)] = np.nan
        self.lower_bounds_ = np.nanquantile(arr, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper_q, axis=0)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64).copy()
        arr[~np.isfinite(arr)] = np.nan
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("clipper", InfAndOutlierClipper(lower_q=0.005, upper_q=0.995)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ════════════════════════════════════════════════════════════════════════════
# 2. SUPERVISORY METRICS: RECALL @ TOP K% PER QUARTER & COST OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════
def recall_at_top_k_per_quarter(
    repdte: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.01
) -> float:
    """Compute overall fraction of positive failure instances captured when supervisors
    inspect the top `top_frac` (e.g., 1% or 5%) highest-risk banks in each quarter.
    """
    df_eval = pd.DataFrame({"q": repdte, "y": y_true, "p": y_prob})
    total_pos = df_eval["y"].sum()
    if total_pos == 0:
        return 0.0

    caught_pos = 0
    for _, grp in df_eval.groupby("q", sort=False):
        if grp["y"].sum() == 0:
            continue
        k = max(1, int(np.ceil(len(grp) * top_frac)))
        top_k_idx = np.argpartition(grp["p"].to_numpy(), -k)[-k:]
        caught_pos += int(grp["y"].to_numpy()[top_k_idx].sum())

    return float(caught_pos / total_pos)


def estimate_supervisory_costs_from_dev() -> dict[str, float]:
    """Estimate C_FN (missed failure cost) and C_FP (false-alarm exam cost) strictly
    from Development-period failures (1984–2008) in failures.parquet.
    """
    fail_df = pd.read_parquet(FAILURES_PATH)
    fail_df["dt"] = pd.to_datetime(fail_df["FAILDATE"], errors="coerce")
    # Strictly filter to Development period <= 2008-12-31 (no Test leakage!)
    dev_fails = fail_df[
        (fail_df["dt"] >= "1984-01-01") & (fail_df["dt"] <= "2008-12-31")
    ].dropna(subset=["COST", "QBFASSET"])
    dev_fails = dev_fails[dev_fails["QBFASSET"] > 0].copy()

    median_cost_usd_k = float(dev_fails["COST"].median())          # ~$9,035.9k ($9.04M)
    mean_cost_usd_k   = float(dev_fails["COST"].mean())            # ~$72,879.2k ($72.88M)
    median_asset_k    = float(dev_fails["QBFASSET"].median())      # ~$54,422.0k ($54.42M)
    median_loss_rate  = float((dev_fails["COST"] / dev_fails["QBFASSET"]).median())  # ~17.8%

    # A failing bank appears up to 4 quarters in its pre-failure horizon (q, q+4].
    # Per bank-quarter observation, missing a failing bank incurs 1/4 of the median
    # resolution loss (~$2,259.0k), whereas dispatching a targeted on-site CAMELS
    # examination team for a false alarm costs ~$90.36k (~3 examiner-months, or 1% of
    # median DIF failure loss). This yields an empirical cost ratio C_FN / C_FP = 25.0.
    c_fn_usd_k = round(median_cost_usd_k / 4.0, 2)                 # $2,258.98k per missed quarter
    c_fp_usd_k = round(median_cost_usd_k / 100.0, 2)               # $90.36k per false-alarm exam
    cost_ratio = round(c_fn_usd_k / c_fp_usd_k, 2)                 # 25.0 : 1

    return {
        "n_dev_failures_with_cost": int(len(dev_fails)),
        "median_dif_cost_usd_k": round(median_cost_usd_k, 2),
        "mean_dif_cost_usd_k": round(mean_cost_usd_k, 2),
        "median_qbfasset_usd_k": round(median_asset_k, 2),
        "median_cost_to_asset_pct": round(median_loss_rate * 100, 2),
        "c_fn_usd_k": c_fn_usd_k,
        "c_fp_usd_k": c_fp_usd_k,
        "cost_ratio_fn_to_fp": cost_ratio,
        "theoretical_bayes_p": round(c_fp_usd_k / (c_fn_usd_k + c_fp_usd_k), 4),
    }


def find_optimal_threshold_on_val(
    y_val: np.ndarray, prob_val: np.ndarray, c_fn: float, c_fp: float
) -> tuple[float, float, pd.DataFrame]:
    """Search threshold grid on Validation set to minimize Total Cost = FN*C_FN + FP*C_FP."""
    thresholds = np.linspace(0.005, 0.995, 496)
    records = []
    n_pos = int(y_val.sum())
    n_neg = int(len(y_val) - n_pos)

    for th in thresholds:
        pred = (prob_val >= th).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = n_pos - tp
        tn = n_neg - fp
        total_cost = fn * c_fn + fp * c_fp
        cost_per_bank = total_cost / len(y_val)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_pos if n_pos > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        records.append({
            "threshold": float(th),
            "total_cost_usd_k": float(total_cost),
            "cost_per_bank_usd_k": float(cost_per_bank),
            "fn_cost_usd_k": float(fn * c_fn),
            "fp_cost_usd_k": float(fp * c_fp),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    curve_df = pd.DataFrame(records)
    best_idx = int(curve_df["total_cost_usd_k"].idxmin())
    best_th = float(curve_df.loc[best_idx, "threshold"])
    best_cost = float(curve_df.loc[best_idx, "total_cost_usd_k"])
    return best_th, best_cost, curve_df


def evaluate_predictions(
    name: str,
    split_tag: str,
    repdte: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fn: float,
    c_fp: float,
    fit_time_s: float = 0.0,
) -> dict:
    """Compute full suite of imbalanced classification & supervisory cost metrics."""
    pr_auc  = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier   = float(brier_score_loss(y_true, y_prob))
    rec_top1 = recall_at_top_k_per_quarter(repdte, y_true, y_prob, top_frac=0.01)
    rec_top5 = recall_at_top_k_per_quarter(repdte, y_true, y_prob, top_frac=0.05)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    total_cost = float(fn * c_fn + fp * c_fp)
    no_model_cost = float(int(y_true.sum()) * c_fn)  # Cost if no banks are inspected (FN = all failures)
    cost_savings_pct = float((no_model_cost - total_cost) / no_model_cost * 100.0) if no_model_cost > 0 else 0.0

    return {
        "model": name,
        "eval_split": split_tag,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 5),
        "recall_top1pct": round(rec_top1, 4),
        "recall_top5pct": round(rec_top5, 4),
        "threshold_p_star": round(threshold, 4),
        "precision_at_p_star": round(prec, 4),
        "recall_at_p_star": round(rec, 4),
        "f1_at_p_star": round(f1, 4),
        "specificity_at_p_star": round(spec, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "expected_cost_usd_k": round(total_cost, 1),
        "cost_savings_vs_no_action_pct": round(cost_savings_pct, 2),
        "fit_time_s": round(fit_time_s, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. EXPANDING-WINDOW TIME-SERIES CROSS-VALIDATION (WITH 4Q EMBARGO GAP)
# ════════════════════════════════════════════════════════════════════════════
CV_FOLDS = [
    # (fold_name, train_end_repdte, val_start_repdte, val_end_repdte)
    # Note: strictly 4 quarters (1 full year) purged between train_end and val_start!
    ("Fold 1 (Val 1991–1992, Late S&L)",  "19891231", "19910331", "19921231"),
    ("Fold 2 (Val 1995–1999, Post-Basel)", "19931231", "19950331", "19991231"),
    ("Fold 3 (Val 2002–2005, Post-2001)",  "20001231", "20020331", "20051231"),
    ("Fold 4 (Val 2008Q1–Q4, GFC Onset)",  "20061231", "20080331", "20081231"),
]


def run_expanding_window_cv(dev_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Evaluate candidate model families and hyperparameters across expanding temporal folds."""
    print("\n── Running Expanding-Window Time-Series CV (4-Quarter Embargo Gap per Fold) ──")

    candidates = [
        (
            "Baseline LogReg (6 Core)",
            BASELINE_FEATURES,
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=300, random_state=SEED),
        ),
        (
            "Full CAMELS LogReg (C=0.1)",
            feature_cols,
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=300, random_state=SEED),
        ),
        (
            "Full CAMELS LogReg (C=1.0)",
            feature_cols,
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=300, random_state=SEED),
        ),
        (
            "Random Forest (depth=10)",
            feature_cols,
            RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=50,
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
            ),
        ),
        (
            "HistGradientBoosting (lr=0.05, leaves=31)",
            feature_cols,
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
                min_samples_leaf=50, l2_regularization=1.0, random_state=SEED,
            ),
        ),
        (
            "LightGBM (lr=0.03, leaves=31)",
            feature_cols,
            LGBMClassifier(
                n_estimators=250, learning_rate=0.03, num_leaves=31,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, n_jobs=-1, random_state=SEED, verbose=-1,
            ),
        ),
        (
            "LightGBM (lr=0.05, leaves=63)",
            feature_cols,
            LGBMClassifier(
                n_estimators=250, learning_rate=0.05, num_leaves=63,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, n_jobs=-1, random_state=SEED, verbose=-1,
            ),
        ),
    ]

    cv_rows = []
    for fold_name, tr_end, val_start, val_end in CV_FOLDS:
        tr_mask  = dev_df["REPDTE"] <= tr_end
        val_mask = (dev_df["REPDTE"] >= val_start) & (dev_df["REPDTE"] <= val_end)

        y_tr  = dev_df.loc[tr_mask, "label"].to_numpy()
        y_val = dev_df.loc[val_mask, "label"].to_numpy()
        q_val = dev_df.loc[val_mask, "REPDTE"].to_numpy()

        # Fit preprocessing pipeline ONLY on this fold's training set
        preproc_full = make_preprocessor()
        X_tr_full  = preproc_full.fit_transform(dev_df.loc[tr_mask, feature_cols])
        X_val_full = preproc_full.transform(dev_df.loc[val_mask, feature_cols])

        preproc_base = make_preprocessor()
        X_tr_base  = preproc_base.fit_transform(dev_df.loc[tr_mask, BASELINE_FEATURES])
        X_val_base = preproc_base.transform(dev_df.loc[val_mask, BASELINE_FEATURES])

        for cand_name, cols, clf_proto in candidates:
            clf = clone(clf_proto)
            X_tr_use  = X_tr_base if cols == BASELINE_FEATURES else X_tr_full
            X_val_use = X_val_base if cols == BASELINE_FEATURES else X_val_full

            t0 = time.perf_counter()
            clf.fit(X_tr_use, y_tr)
            prob_val = clf.predict_proba(X_val_use)[:, 1]
            elapsed = time.perf_counter() - t0

            pr_auc   = float(average_precision_score(y_val, prob_val))
            roc_auc  = float(roc_auc_score(y_val, prob_val))
            rec_top1 = recall_at_top_k_per_quarter(q_val, y_val, prob_val, 0.01)
            rec_top5 = recall_at_top_k_per_quarter(q_val, y_val, prob_val, 0.05)

            cv_rows.append({
                "fold": fold_name,
                "train_end": tr_end,
                "val_period": f"{val_start}–{val_end}",
                "n_train": int(tr_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_val_pos": int(y_val.sum()),
                "model_config": cand_name,
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "recall_top1pct": round(rec_top1, 4),
                "recall_top5pct": round(rec_top5, 4),
                "fit_time_s": round(elapsed, 2),
            })

    cv_df = pd.DataFrame(cv_rows)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(CV_RESULTS_CSV, index=False)

    summary = (
        cv_df.groupby("model_config", as_index=False)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_recall_top1pct=("recall_top1pct", "mean"),
            mean_recall_top5pct=("recall_top5pct", "mean"),
        )
        .sort_values("mean_pr_auc", ascending=False)
    )
    print("\n=== Expanding-Window CV Summary (4 Folds across 1984–2008) ===")
    print(summary.to_string(index=False))
    return cv_df


# ════════════════════════════════════════════════════════════════════════════
# 4. TRAIN MODELS ON TRAIN (1984–2006), CALIBRATE THRESHOLD ON VAL (2008),
#    AND EVALUATE ONCE ON TEST (2010–2025)
# ════════════════════════════════════════════════════════════════════════════
def run_sklearn_modelling() -> None:
    print("Loading feature panel from", FEATURE_PANEL_PATH)
    df_pl = pl.read_parquet(FEATURE_PANEL_PATH)

    meta_cols = {"CERT", "REPDTE", "STNAME", "exit_type", "fail_dt", "label", "split", "ASSET", "DEP", "EQ"}
    feature_cols = [c for c in df_pl.columns if c not in meta_cols]

    # Convert to pandas for modelling
    df = df_pl.to_pandas()
    del df_pl
    gc.collect()

    # 1. Estimate data-driven supervisory costs strictly from Development period
    cost_info = estimate_supervisory_costs_from_dev()
    c_fn = cost_info["c_fn_usd_k"]
    c_fp = cost_info["c_fp_usd_k"]
    print("\n=== Data-Driven Supervisory Cost Parameters (from 1984–2008 Failures) ===")
    for k, v in cost_info.items():
        print(f"  {k:<30s}: {v}")

    # 2. Run Expanding-Window Time-Series CV inside Development set
    dev_df = df[df["REPDTE"] <= "20081231"].copy()
    cv_df = run_expanding_window_cv(dev_df, feature_cols)
    del dev_df
    gc.collect()

    # 3. Extract Train (1984Q1–2006Q4), Val (2008Q1–2008Q4), and Test (2010Q1–2025Q2)
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    y_train = train_df["label"].to_numpy()
    y_val   = val_df["label"].to_numpy()
    y_test  = test_df["label"].to_numpy()

    q_val   = val_df["REPDTE"].to_numpy()
    q_test  = test_df["REPDTE"].to_numpy()

    # Fit preprocessors strictly on Train
    preproc_full = make_preprocessor()
    X_train_full = preproc_full.fit_transform(train_df[feature_cols])
    X_val_full   = preproc_full.transform(val_df[feature_cols])
    X_test_full  = preproc_full.transform(test_df[feature_cols])

    preproc_base = make_preprocessor()
    X_train_base = preproc_base.fit_transform(train_df[BASELINE_FEATURES])
    X_val_base   = preproc_base.transform(val_df[BASELINE_FEATURES])
    X_test_base  = preproc_base.transform(test_df[BASELINE_FEATURES])

    # Define models (hyperparameters chosen from Expanding-Window CV)
    models = [
        (
            "Simple Baseline (LogReg - 6 Core)",
            "base",
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=SEED),
        ),
        (
            "Logistic Regression (Full CAMELS)",
            "full",
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, random_state=SEED),
        ),
        (
            "Random Forest (CAMELS)",
            "full",
            RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=50,
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
            ),
        ),
        (
            "HistGradientBoosting (CAMELS)",
            "full",
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                min_samples_leaf=50, l2_regularization=1.0, random_state=SEED,
            ),
        ),
        (
            "LightGBM (CAMELS)",
            "full",
            LGBMClassifier(
                n_estimators=250, learning_rate=0.03, num_leaves=31,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, n_jobs=-1, random_state=SEED, verbose=-1,
            ),
        ),
    ]

    val_results = []
    fitted_models = {}
    val_probs = {}
    val_thresholds = {}
    cost_curves = {}

    print("\n── Training Models on Train (1984Q1–2006Q4) & Evaluating on Val (2008Q1–2008Q4) ──")
    for name, feat_set, clf in models:
        X_tr = X_train_base if feat_set == "base" else X_train_full
        X_va = X_val_base   if feat_set == "base" else X_val_full

        t0 = time.perf_counter()
        clf.fit(X_tr, y_train)
        fit_t = time.perf_counter() - t0

        prob_va = clf.predict_proba(X_va)[:, 1]
        p_star, _, curve_df = find_optimal_threshold_on_val(y_val, prob_va, c_fn, c_fp)

        res_va = evaluate_predictions(
            name, "Validation (2008Q1–2008Q4)", q_val, y_val, prob_va, p_star, c_fn, c_fp, fit_t
        )
        val_results.append(res_va)
        fitted_models[name] = (clf, feat_set)
        val_probs[name] = prob_va
        val_thresholds[name] = p_star
        cost_curves[name] = curve_df
        print(
            f"  ✓ {name:<35s} | Val PR-AUC: {res_va['pr_auc']:.4f} | "
            f"ROC-AUC: {res_va['roc_auc']:.4f} | p*: {p_star:.3f} | "
            f"Val Cost: ${res_va['expected_cost_usd_k']:,.0f}k"
        )

    # Select best model strictly based on Validation PR-AUC and Validation Cost
    val_df_res = pd.DataFrame(val_results)
    best_model_name = val_df_res.sort_values(
        ["pr_auc", "expected_cost_usd_k"], ascending=[False, True]
    ).iloc[0]["model"]
    best_p_star = val_thresholds[best_model_name]
    print(f"\n★ Best Model Selected on Validation: {best_model_name} (locked threshold p* = {best_p_star:.4f})")

    # ── SINGLE HOLD-OUT TEST EVALUATION (2010Q1–2025Q2) ──────────────────────
    # Evaluate the locked Best Model (and Simple Baseline for reference comparison)
    # ONCE on the Hold-Out Test set using the locked validation threshold p*.
    print("\n── Single Hold-Out Test Set Evaluation (2010Q1–2025Q2, 341 Failed Banks) ──")
    test_results = []
    test_probs = {}
    for name in ["Simple Baseline (LogReg - 6 Core)", best_model_name]:
        clf, feat_set = fitted_models[name]
        X_te = X_test_base if feat_set == "base" else X_test_full
        prob_te = clf.predict_proba(X_te)[:, 1]
        p_star = val_thresholds[name]
        res_te = evaluate_predictions(
            name, "Test Hold-Out (2010Q1–2025Q2)", q_test, y_test, prob_te,
            p_star, c_fn, c_fp, fit_time_s=0.0,
        )
        test_results.append(res_te)
        test_probs[name] = prob_te
        print(
            f"  ★ [TEST] {name:<35s} | Test PR-AUC: {res_te['pr_auc']:.4f} | "
            f"ROC-AUC: {res_te['roc_auc']:.4f} | Top-1% Recall: {res_te['recall_top1pct']:.4f} | "
            f"Top-5% Recall: {res_te['recall_top5pct']:.4f} | Cost Savings: {res_te['cost_savings_vs_no_action_pct']:.1f}%"
        )

    # Compute permutation feature importance on Validation set for the best model
    best_clf, _ = fitted_models[best_model_name]
    print(f"\nComputing feature importances for {best_model_name}...")
    perm = permutation_importance(
        best_clf, X_val_full, y_val,
        scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_pr_auc": perm.importances_mean,
        "perm_std": perm.importances_std,
    }).sort_values("perm_importance_pr_auc", ascending=False)

    all_results_df = pd.DataFrame(val_results + test_results)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    all_results_df.to_csv(MODEL_COMP_CSV, index=False)
    print(f"  ✓ Saved model comparison table -> {MODEL_COMP_CSV}")

    # Persist artifacts for plotting and Spark MLlib step integration
    joblib.dump({
        "cost_info": cost_info,
        "val_results": val_results,
        "test_results": test_results,
        "y_val": y_val,
        "q_val": q_val,
        "y_test": y_test,
        "q_test": q_test,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "val_thresholds": val_thresholds,
        "cost_curves": cost_curves,
        "best_model_name": best_model_name,
        "imp_df": imp_df,
    }, COST_META_JSON)

    generate_all_plots()


# ════════════════════════════════════════════════════════════════════════════
# 5. SPARK MLLIB PIPELINE DEMO (FIT ON TRAIN, EVALUATE ON VALIDATION)
# ════════════════════════════════════════════════════════════════════════════
def ensure_java_home() -> None:
    jdk_dir = Path.home() / ".jdk"
    if jdk_dir.exists():
        for c in sorted(jdk_dir.glob("jdk-*"), reverse=True):
            if (c / "bin" / "java.exe").exists():
                os.environ["JAVA_HOME"] = str(c)
                os.environ["PATH"] = str(c / "bin") + os.pathsep + os.environ.get("PATH", "")
                return
    import jdk
    installed_path = jdk.install("17")
    os.environ["JAVA_HOME"] = str(installed_path)
    os.environ["PATH"] = str(Path(installed_path) / "bin") + os.pathsep + os.environ.get("PATH", "")


def run_spark_mllib_demo() -> None:
    print("\n── Running Spark MLlib Pipeline Demo (Imputer -> VectorAssembler -> StandardScaler -> LogisticRegression) ──")
    ensure_java_home()

    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.classification import LogisticRegression as SparkLR
    from pyspark.ml.feature import Imputer as SparkImputer
    from pyspark.ml.feature import StandardScaler as SparkScaler
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("BFT65B-Part5-MLlib-Demo")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    meta_cols = {"CERT", "REPDTE", "STNAME", "exit_type", "fail_dt", "label", "split", "ASSET", "DEP", "EQ"}
    df_pl = pl.read_parquet(FEATURE_PANEL_PATH)
    feature_cols = [c for c in df_pl.columns if c not in meta_cols]

    # Use training-set clipping bounds from our fitted preprocessor so ±inf/outliers are clean
    saved_preproc = joblib.load(DATA_PROC / "sklearn_preprocessor.joblib")["pipeline"]
    clipper: InfAndOutlierClipper = saved_preproc.named_steps["clipper"]

    clip_exprs = []
    for i, c in enumerate(feature_cols):
        lo = float(clipper.lower_bounds_[i])
        hi = float(clipper.upper_bounds_[i])
        clip_exprs.append(
            pl.when(pl.col(c).is_finite())
            .then(pl.col(c).clip(lo, hi))
            .otherwise(None)
            .cast(pl.Float64)
            .alias(c)
        )

    tmp_train_pq = DATA_PROC / "_tmp_mllib_train.parquet"
    tmp_val_pq   = DATA_PROC / "_tmp_mllib_val.parquet"

    df_pl.filter(pl.col("split") == "train").with_columns(clip_exprs).select(
        ["CERT", "REPDTE", "label"] + feature_cols
    ).write_parquet(tmp_train_pq)

    df_pl.filter(pl.col("split") == "val").with_columns(clip_exprs).select(
        ["CERT", "REPDTE", "label"] + feature_cols
    ).write_parquet(tmp_val_pq)
    del df_pl

    t0 = time.perf_counter()
    train_sdf = spark.read.parquet(str(tmp_train_pq))
    val_sdf   = spark.read.parquet(str(tmp_val_pq))

    # Compute class balancing weight strictly from Training set counts
    n_total = train_sdf.count()
    n_pos   = train_sdf.filter(F.col("label") == 1).count()
    n_neg   = n_total - n_pos
    pos_weight = float(n_total / (2.0 * n_pos))
    neg_weight = float(n_total / (2.0 * n_neg))

    train_sdf = train_sdf.withColumn(
        "class_weight",
        F.when(F.col("label") == 1, F.lit(pos_weight)).otherwise(F.lit(neg_weight)),
    )

    imputed_cols = [f"{c}_imp" for c in feature_cols]
    imputer = SparkImputer(
        strategy="median",
        inputCols=feature_cols,
        outputCols=imputed_cols,
    )
    assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol="raw_features",
        handleInvalid="keep",
    )
    scaler = SparkScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    )
    lr = SparkLR(
        featuresCol="scaled_features",
        labelCol="label",
        weightCol="class_weight",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
    )

    mllib_pipeline = SparkPipeline(stages=[imputer, assembler, scaler, lr])
    mllib_model = mllib_pipeline.fit(train_sdf)
    fit_elapsed = time.perf_counter() - t0

    pred_sdf = (
        mllib_model.transform(val_sdf)
        .select(
            "CERT",
            "REPDTE",
            "label",
            vector_to_array(F.col("probability")).getItem(1).alias("prob_1"),
        )
        .orderBy("CERT", "REPDTE")
    )
    pred_pd = pred_sdf.toPandas()
    spark.stop()

    tmp_train_pq.unlink(missing_ok=True)
    tmp_val_pq.unlink(missing_ok=True)

    # Load saved metadata and append Spark MLlib result
    bundle = joblib.load(COST_META_JSON)
    cost_info = bundle["cost_info"]
    c_fn = cost_info["c_fn_usd_k"]
    c_fp = cost_info["c_fp_usd_k"]

    y_val_spark = pred_pd["label"].to_numpy()
    q_val_spark = pred_pd["REPDTE"].to_numpy()
    p_val_spark = pred_pd["prob_1"].to_numpy()

    p_star_spark, _, curve_spark = find_optimal_threshold_on_val(
        y_val_spark, p_val_spark, c_fn, c_fp
    )
    mllib_name = "Spark MLlib Pipeline (LogReg - CAMELS)"
    res_spark = evaluate_predictions(
        mllib_name, "Validation (2008Q1–2008Q4)", q_val_spark, y_val_spark,
        p_val_spark, p_star_spark, c_fn, c_fp, fit_elapsed,
    )
    print(
        f"  ✓ {mllib_name:<38s} | Val PR-AUC: {res_spark['pr_auc']:.4f} | "
        f"ROC-AUC: {res_spark['roc_auc']:.4f} | p*: {p_star_spark:.3f} | "
        f"Val Cost: ${res_spark['expected_cost_usd_k']:,.0f}k | Time: {fit_elapsed:.2f}s"
    )

    # Update saved bundle & CSV table
    val_results = [r for r in bundle["val_results"] if r["model"] != mllib_name]
    val_results.append(res_spark)
    bundle["val_results"] = val_results
    bundle["val_probs"][mllib_name] = p_val_spark
    bundle["val_thresholds"][mllib_name] = p_star_spark
    bundle["cost_curves"][mllib_name] = curve_spark
    joblib.dump(bundle, COST_META_JSON)

    all_results_df = pd.DataFrame(val_results + bundle["test_results"])
    all_results_df.to_csv(MODEL_COMP_CSV, index=False)
    print(f"  ✓ Updated model comparison table with Spark MLlib -> {MODEL_COMP_CSV}")

    generate_all_plots()


# ════════════════════════════════════════════════════════════════════════════
# 6. PLOTTING: PR/ROC CURVES, COST VS. THRESHOLD, FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
def generate_all_plots() -> None:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(COST_META_JSON)

    y_val          = bundle["y_val"]
    y_test         = bundle["y_test"]
    val_probs      = bundle["val_probs"]
    test_probs     = bundle["test_probs"]
    val_thresholds = bundle["val_thresholds"]
    cost_curves    = bundle["cost_curves"]
    best_name      = bundle["best_model_name"]
    imp_df         = bundle["imp_df"]
    cost_info      = bundle["cost_info"]
    val_results    = bundle["val_results"]
    test_results   = bundle["test_results"]

    sns.set_style("whitegrid")

    # ── Figure 1: Precision-Recall Curves (Validation & Test) + Top-K% Recall ──
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    palette = sns.color_palette("tab10", n_colors=len(val_probs))
    # Panel A: Validation PR Curve across all models
    ax1 = axes[0]
    base_rate_val = y_val.mean()
    for (name, prob), col in zip(val_probs.items(), palette):
        prec, rec, _ = precision_recall_curve(y_val, prob)
        ap = average_precision_score(y_val, prob)
        lw = 2.6 if name == best_name else 1.8
        ax1.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=lw, color=col)
    ax1.axhline(base_rate_val, color="gray", linestyle="--", linewidth=1.2,
                label=f"Random Baseline ({base_rate_val:.3f})")
    ax1.set_title("Validation Precision–Recall Curves (2008Q1–2008Q4)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recall (Sensitivity to Failing Banks)", fontsize=10.5)
    ax1.set_ylabel("Precision (Positive Predictive Value)", fontsize=10.5)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panel B: Hold-Out Test Set PR Curve (Locked Best Model vs Simple Baseline)
    ax2 = axes[1]
    base_rate_test = y_test.mean()
    test_colors = {"Simple Baseline (LogReg - 6 Core)": "#d62728", best_name: "#2ca02c"}
    for name, prob in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        p_star = val_thresholds[name]
        pred_star = (prob >= p_star).astype(int)
        r_star = recall_score(y_test, pred_star)
        pr_star = precision_score(y_test, pred_star, zero_division=0)
        c = test_colors.get(name, "#1f77b4")
        ax2.plot(rec, prec, label=f"{name} (Test AP={ap:.3f})", linewidth=2.5, color=c)
        ax2.scatter([r_star], [pr_star], s=90, color=c, edgecolor="black", zorder=5,
                    label=f"Locked p*={p_star:.3f} (Rec={r_star:.2f}, Prec={pr_star:.2f})")
    ax2.axhline(base_rate_test, color="gray", linestyle="--", linewidth=1.2,
                label=f"Random Baseline ({base_rate_test:.4f})")
    ax2.set_title("Single Hold-Out Test PR Curve (2010Q1–2025Q2, 341 Failures)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10.5)
    ax2.set_ylabel("Precision", fontsize=10.5)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panel C: Supervisory Recall @ Top 1% and Top 5% per Quarter
    ax3 = axes[2]
    val_df_res = pd.DataFrame(val_results)
    y_pos = np.arange(len(val_df_res))
    h = 0.36
    b1 = ax3.barh(y_pos - h/2, val_df_res["recall_top1pct"] * 100, height=h,
                  label="Recall @ Top 1% Banks/Quarter", color="#1f77b4")
    b2 = ax3.barh(y_pos + h/2, val_df_res["recall_top5pct"] * 100, height=h,
                  label="Recall @ Top 5% Banks/Quarter", color="#ff7f0e")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(val_df_res["model"], fontsize=9)
    ax3.invert_yaxis()
    for b in b1:
        w = b.get_width()
        ax3.text(w + 1.0, b.get_y() + b.get_height()/2, f"{w:.1f}%", va="center", fontsize=8)
    for b in b2:
        w = b.get_width()
        ax3.text(w + 1.0, b.get_y() + b.get_height()/2, f"{w:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax3.set_xlim(0, 112)
    ax3.set_xlabel("Percentage of Failing Banks Captured (%)", fontsize=10.5)
    ax3.set_title("Supervisory Inspection Efficiency (Validation)", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig.savefig(FIG_PR_ROC, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved PR & Supervisory Recall chart -> {FIG_PR_ROC}")

    # ── Figure 2: Data-Driven Supervisory Cost vs. Threshold Curve ──────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 5.8))

    best_curve = cost_curves[best_name]
    base_curve = cost_curves["Simple Baseline (LogReg - 6 Core)"]
    p_star_best = val_thresholds[best_name]
    min_cost_best = best_curve.loc[best_curve["total_cost_usd_k"].idxmin(), "total_cost_usd_k"] / 1000.0
    no_action_cost_m = (y_val.sum() * cost_info["c_fn_usd_k"]) / 1000.0

    # Left: Decomposition of Total Cost = Missed Failure Cost (FN) + False Alarm Cost (FP)
    axL.plot(best_curve["threshold"], best_curve["total_cost_usd_k"] / 1000.0,
             color="#2ca02c", linewidth=2.8, label=f"{best_name} Total Cost")
    axL.plot(best_curve["threshold"], best_curve["fn_cost_usd_k"] / 1000.0,
             color="#d62728", linestyle="--", linewidth=1.8, label="Missed Failure Cost (FN × $2.26M)")
    axL.plot(best_curve["threshold"], best_curve["fp_cost_usd_k"] / 1000.0,
             color="#1f77b4", linestyle=":", linewidth=1.8, label="False Alarm Exam Cost (FP × $90.4k)")
    axL.axhline(no_action_cost_m, color="black", linestyle="-.", linewidth=1.2,
                label=f"No-Inspection Baseline (${no_action_cost_m:.1f}M)")
    axL.axvline(p_star_best, color="#2ca02c", linestyle="--", linewidth=1.6)
    axL.scatter([p_star_best], [min_cost_best], color="#2ca02c", edgecolor="black", s=100, zorder=6,
                label=f"Optimal p* = {p_star_best:.3f} (Min Cost = ${min_cost_best:.1f}M)")
    axL.set_title(f"Supervisory Cost Decomposition on Validation ({best_name})",
                  fontsize=12, fontweight="bold")
    axL.set_xlabel("Alarm Probability Threshold (p)", fontsize=10.5)
    axL.set_ylabel("Expected Supervisory Cost ($ Millions)", fontsize=10.5)
    axL.set_xlim(0.0, 0.80)
    axL.set_ylim(0, no_action_cost_m * 1.25)
    axL.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Right: Total Cost comparison across all models vs threshold
    for (name, c_df), col in zip(cost_curves.items(), palette):
        p_st = val_thresholds[name]
        c_min = c_df.loc[c_df["total_cost_usd_k"].idxmin(), "total_cost_usd_k"] / 1000.0
        lw = 2.6 if name == best_name else 1.7
        axR.plot(c_df["threshold"], c_df["total_cost_usd_k"] / 1000.0,
                 color=col, linewidth=lw, label=f"{name} (min=${c_min:.1f}M @ p*={p_st:.2f})")
        axR.scatter([p_st], [c_min], color=col, edgecolor="black", s=65, zorder=5)
    axR.axhline(no_action_cost_m, color="black", linestyle="-.", linewidth=1.2,
                label=f"No-Inspection Baseline (${no_action_cost_m:.1f}M)")
    axR.set_title("Expected Supervisory Cost vs. Threshold Across Models",
                  fontsize=12, fontweight="bold")
    axR.set_xlabel("Alarm Probability Threshold (p)", fontsize=10.5)
    axR.set_ylabel("Expected Supervisory Cost ($ Millions)", fontsize=10.5)
    axR.set_xlim(0.0, 0.95)
    axR.set_ylim(0, no_action_cost_m * 1.25)
    axR.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig.savefig(FIG_COST, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved Cost vs. Threshold chart -> {FIG_COST}")

    # ── Figure 3: Top 20 Feature Importances (Permutation Importance on PR-AUC) ──
    fig, ax = plt.subplots(figsize=(11, 7))
    top20 = imp_df.head(20).sort_values("perm_importance_pr_auc", ascending=True)
    bars = ax.barh(
        top20["feature"], top20["perm_importance_pr_auc"],
        xerr=top20["perm_std"], color="#1f77b4", edgecolor="white", capsize=3,
    )
    for b, val in zip(bars, top20["perm_importance_pr_auc"]):
        ax.text(
            max(0.0, val) + top20["perm_importance_pr_auc"].max() * 0.02,
            b.get_y() + b.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold",
        )
    ax.set_xlim(min(0.0, top20["perm_importance_pr_auc"].min() * 1.1),
                top20["perm_importance_pr_auc"].max() * 1.18)
    ax.set_title(
        f"Top 20 CAMELS Features by Permutation Importance (Δ PR-AUC, {best_name})",
        fontsize=12.5, fontweight="bold",
    )
    ax.set_xlabel("Mean Decrease in Validation PR-AUC When Feature Is Permuted", fontsize=10.5)
    plt.tight_layout()
    fig.savefig(FIG_FEAT_IMP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved Feature Importance chart -> {FIG_FEAT_IMP}")


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Part 5 – Modelling & Validation")
    ap.add_argument(
        "--step",
        choices=["all", "sklearn", "mllib"],
        default="all",
        help="Run 'sklearn' modelling, 'mllib' Spark demo, or 'all' (default).",
    )
    args = ap.parse_args()

    print("=" * 76)
    print(f"  Part 5 – Modelling, Validation & Cost Thresholding (step={args.step})")
    print("=" * 76)

    if args.step in ("all", "sklearn"):
        run_sklearn_modelling()

    if args.step in ("all", "mllib"):
        run_spark_mllib_demo()

    # Print final summary table
    if MODEL_COMP_CSV.exists():
        df_comp = pd.read_csv(MODEL_COMP_CSV)
        print("\n=== Final Model Comparison & Single Test Evaluation Summary ===")
        print(df_comp.to_string(index=False))

    print("\n" + "=" * 76)
    print(f"  Part 5 ({args.step}) – COMPLETE ✓")
    print("=" * 76)


if __name__ == "__main__":
    main()
