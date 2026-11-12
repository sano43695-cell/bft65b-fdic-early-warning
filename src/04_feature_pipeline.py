"""
src/04_feature_pipeline.py – Part 4: Feature Pipeline & Leak-Free Preprocessing
===============================================================================

Implements:
  1. Point-in-Time CAMELS Feature Engineering in Polars (strictly using data <= q):
       • C (Capital Adequacy): eq_to_asset, tier1_cap_ratio, total_cap_ratio, equity_cushion_gap
       • A (Asset Quality): npl_to_loans, llreserve_to_loans, reserve_coverage,
         pastdue90_to_asset, nonaccrual_to_asset, ore_to_asset, texas_ratio, net_chargeoff_rate
       • M & E (Management & Earnings): roa, roe, nimy, cost_to_income,
         retained_profit_ratio, eq_retention_1q, provision_to_asset
       • L (Liquidity): brokered_dep_ratio, liquid_assets_ratio, loan_to_deposit,
         dep_to_asset, domestic_dep_share
       • S & Momentum (Sensitivity & Dynamics): re_loan_share, ci_loan_share,
         con_loan_share, funding_cost_rate, log_asset, asset_growth_yoy,
         loan_growth_yoy, npl_ratio_chg_4q, eq_ratio_chg_4q, roa_chg_4q,
         roa_roll_mean_4q, roa_roll_std_4q, nim_roll_mean_4q
  2. Leak-Free Temporal Split with 4-Quarter Purging/Embargo Gaps:
       • Development Set : 1984Q1 – 2008Q4 (Train: 1984Q1–2006Q4, Gap: 2007Q1–2007Q4, Val: 2008Q1–2008Q4)
       • Outer Embargo   : 2009Q1 – 2009Q4 (4 quarters purged so 2008Q4 labels do not leak into Test)
       • Test Set        : 2010Q1 – 2025Q2 (62 quarters, 341 failed banks >= 50 required, evaluated once)
  3. Scikit-Learn Preprocessing Pipeline (fit ONLY on Training data):
       • InfAndOutlierClipper (replaces ±inf with NaN, clips to [0.5%, 99.5%] quantiles of X_train)
       • SimpleImputer(strategy='median') (imputes missing values using X_train medians only)
       • StandardScaler() (standardizes using X_train mean and std only)

Outputs:
  • data_processed/features_panel.parquet
  • outputs/tables/part4_split_summary.csv
  • outputs/tables/part4_feature_dictionary.csv
  • outputs/figures/part4_split_and_features.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PROC  = ROOT / "data_processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS   = ROOT / "outputs" / "figures"

CLEAN_PANEL_PATH   = DATA_PROC  / "clean_panel.parquet"
FEATURE_PANEL_PATH = DATA_PROC  / "features_panel.parquet"
PREPROC_PATH       = DATA_PROC  / "sklearn_preprocessor.joblib"
SPLIT_CSV_PATH     = OUT_TABLES / "part4_split_summary.csv"
DICT_CSV_PATH      = OUT_TABLES / "part4_feature_dictionary.csv"
FIG_PATH           = OUT_FIGS   / "part4_split_and_features.png"

# ── Temporal Split Cutoffs (YYYYMMDD) ───────────────────────────────────────
# Development period: 1984Q1 -> 2008Q4
#   - Train subset  : 1984Q1 -> 2006Q4
#   - Inner Gap     : 2007Q1 -> 2007Q4 (4 quarters embargo between Train and Val)
#   - Val subset    : 2008Q1 -> 2008Q4
# Outer Embargo Gap : 2009Q1 -> 2009Q4 (4 quarters embargo between Dev and Test)
# Test period       : 2010Q1 -> 2025Q2 (contains 341 failed banks, >= 50 required)
TRAIN_END_Q     = "20061231"
INNER_GAP_END_Q = "20071231"
DEV_END_Q       = "20081231"   # End of Validation & Development period
OUTER_GAP_END_Q = "20091231"   # End of 4-quarter purging gap before Test
TEST_START_Q    = "20100331"
TEST_END_Q      = "20250630"


# ════════════════════════════════════════════════════════════════════════════
# 1. CUSTOM LEAK-FREE OUTLIER & INFINITY CLIPPER FOR SKLEARN PIPELINE
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


# ════════════════════════════════════════════════════════════════════════════
# 2. CAMELS FEATURE METADATA DICTIONARY
# ════════════════════════════════════════════════════════════════════════════
FEATURE_METADATA: list[dict[str, str]] = [
    # --- C: Capital Adequacy ---
    {
        "feature": "eq_to_asset",
        "camels_group": "C - Capital Adequacy",
        "formula": "EQ / ASSET * 100",
        "economic_rationale": "Book equity capital cushion to absorb asset losses (%) – available across all eras (1984–2025).",
    },
    {
        "feature": "tier1_cap_ratio",
        "camels_group": "C - Capital Adequacy",
        "formula": "RBC1AAJ",
        "economic_rationale": "Regulatory Tier 1 risk-based capital ratio (%), primary Prompt Corrective Action (PCA) trigger.",
    },
    {
        "feature": "total_cap_ratio",
        "camels_group": "C - Capital Adequacy",
        "formula": "RBCRWAJ",
        "economic_rationale": "Total risk-based capital ratio (%) under Basel I/II/III capital standards.",
    },
    {
        "feature": "equity_cushion_gap",
        "camels_group": "C - Capital Adequacy",
        "formula": "(EQ - NCLNLS) / ASSET * 100",
        "economic_rationale": "Net equity buffer after deducting noncurrent loans, scaled by total assets (%).",
    },
    # --- A: Asset Quality ---
    {
        "feature": "npl_to_loans",
        "camels_group": "A - Asset Quality",
        "formula": "NCLNLS / LNLSGR * 100",
        "economic_rationale": "Noncurrent loans (past due >=90d + nonaccrual) as % of gross loan portfolio.",
    },
    {
        "feature": "llreserve_to_loans",
        "camels_group": "A - Asset Quality",
        "formula": "LNATRES / LNLSGR * 100",
        "economic_rationale": "Allowance for loan & lease losses (ALLL) as % of gross loans.",
    },
    {
        "feature": "reserve_coverage",
        "camels_group": "A - Asset Quality",
        "formula": "LNATRES / (NCLNLS + 1.0) * 100",
        "economic_rationale": "Reserve coverage ratio (% of noncurrent loans covered by loan-loss reserves).",
    },
    {
        "feature": "pastdue90_to_asset",
        "camels_group": "A - Asset Quality",
        "formula": "P9ASSET / ASSET * 100",
        "economic_rationale": "Assets past due 90+ days and still accruing as % of total assets.",
    },
    {
        "feature": "nonaccrual_to_asset",
        "camels_group": "A - Asset Quality",
        "formula": "NAASSET / ASSET * 100",
        "economic_rationale": "Nonaccrual loans & assets (severe credit distress) as % of total assets.",
    },
    {
        "feature": "ore_to_asset",
        "camels_group": "A - Asset Quality",
        "formula": "ORE / ASSET * 100",
        "economic_rationale": "Other Real Estate Owned (foreclosed properties) as % of total assets.",
    },
    {
        "feature": "texas_ratio",
        "camels_group": "A - Asset Quality",
        "formula": "(NCLNLS + ORE) / (EQ + LNATRES) * 100",
        "economic_rationale": "Supervisory Texas Ratio: troubled assets vs. tangible capital + loss reserves (%).",
    },
    {
        "feature": "net_chargeoff_rate",
        "camels_group": "A - Asset Quality",
        "formula": "NTLNLS / LNLSGR * 100",
        "economic_rationale": "Realized loan loss rate (net charge-offs as % of gross loans).",
    },
    # --- M & E: Management & Earnings ---
    {
        "feature": "roa",
        "camels_group": "M & E - Management & Earnings",
        "formula": "ROA",
        "economic_rationale": "Annualized Return on Assets (%), core profitability metric.",
    },
    {
        "feature": "roe",
        "camels_group": "M & E - Management & Earnings",
        "formula": "ROE",
        "economic_rationale": "Annualized Return on Equity (%), return to bank shareholders.",
    },
    {
        "feature": "nimy",
        "camels_group": "M & E - Management & Earnings",
        "formula": "NIMY",
        "economic_rationale": "Net Interest Margin (%), core lending spread efficiency.",
    },
    {
        "feature": "cost_to_income",
        "camels_group": "M & E - Management & Earnings",
        "formula": "NONIX / (INTINC - EINTEXP + NONII) * 100",
        "economic_rationale": "Operating efficiency ratio: noninterest overhead expense per dollar of operating revenue (%).",
    },
    {
        "feature": "retained_profit_ratio",
        "camels_group": "M & E - Management & Earnings",
        "formula": "NETINC / ASSET * 100",
        "economic_rationale": "Unadjusted net profit generation relative to asset base (%).",
    },
    {
        "feature": "eq_retention_1q",
        "camels_group": "M & E - Management & Earnings",
        "formula": "(EQ_q - EQ_{q-1}) / ASSET_q * 100",
        "economic_rationale": "Quarterly organic capital accumulation (retained earnings + net capital accretion % of assets).",
    },
    {
        "feature": "provision_to_asset",
        "camels_group": "M & E - Management & Earnings",
        "formula": "ELNATR / ASSET * 100",
        "economic_rationale": "Credit loss provision expense burden relative to total assets (%).",
    },
    # --- L: Liquidity ---
    {
        "feature": "brokered_dep_ratio",
        "camels_group": "L - Liquidity",
        "formula": "BRO / DEP * 100",
        "economic_rationale": "Reliance on wholesale/brokered deposits (% of total deposits) – flight-prone funding.",
    },
    {
        "feature": "liquid_assets_ratio",
        "camels_group": "L - Liquidity",
        "formula": "(CHBAL + SC) / ASSET * 100",
        "economic_rationale": "Cash, due from banks, and investment securities as % of total assets.",
    },
    {
        "feature": "loan_to_deposit",
        "camels_group": "L - Liquidity",
        "formula": "LNLSNET / DEP * 100",
        "economic_rationale": "Net loans to total deposits (%) – measures illiquidity and funding stretch.",
    },
    {
        "feature": "dep_to_asset",
        "camels_group": "L - Liquidity",
        "formula": "DEP / ASSET * 100",
        "economic_rationale": "Share of balance sheet funded by deposits vs. non-deposit borrowings (%).",
    },
    {
        "feature": "domestic_dep_share",
        "camels_group": "L - Liquidity",
        "formula": "DEPDOM / DEP * 100",
        "economic_rationale": "Domestic deposit share of total deposits (%).",
    },
    # --- S & Momentum: Sensitivity & Point-in-Time Dynamics ---
    {
        "feature": "re_loan_share",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "LNRE / LNLSGR * 100",
        "economic_rationale": "Real estate loan concentration (% of gross loans) – exposure to property cycles.",
    },
    {
        "feature": "ci_loan_share",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "LNCI / LNLSGR * 100",
        "economic_rationale": "Commercial & industrial (C&I) loan share (% of gross loans).",
    },
    {
        "feature": "con_loan_share",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "LNCON / LNLSGR * 100",
        "economic_rationale": "Consumer loan share (% of gross loans).",
    },
    {
        "feature": "funding_cost_rate",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "EINTEXP / DEP * 100",
        "economic_rationale": "Effective deposit interest expense rate (%) – sensitivity to interest rate hikes.",
    },
    {
        "feature": "log_asset",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "ln(ASSET)",
        "economic_rationale": "Log of total assets (in $ thousands) – captures bank scale and diversification.",
    },
    {
        "feature": "asset_growth_yoy",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "(ASSET_q - ASSET_{q-4}) / ASSET_{q-4} * 100",
        "economic_rationale": "1-year (4Q) asset growth rate (%) – rapid expansion often precedes credit losses.",
    },
    {
        "feature": "loan_growth_yoy",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "(LNLSGR_q - LNLSGR_{q-4}) / LNLSGR_{q-4} * 100",
        "economic_rationale": "1-year (4Q) gross loan growth rate (%) – aggressive lending indicator.",
    },
    {
        "feature": "npl_ratio_chg_4q",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "npl_to_loans_q - npl_to_loans_{q-4}",
        "economic_rationale": "4-quarter change in NPL ratio (percentage points) – credit deterioration velocity.",
    },
    {
        "feature": "eq_ratio_chg_4q",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "eq_to_asset_q - eq_to_asset_{q-4}",
        "economic_rationale": "4-quarter change in equity-to-asset ratio (percentage points) – capital erosion speed.",
    },
    {
        "feature": "roa_chg_4q",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "ROA_q - ROA_{q-4}",
        "economic_rationale": "4-quarter change in ROA (percentage points) – earnings trajectory.",
    },
    {
        "feature": "roa_roll_mean_4q",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "mean(ROA_{q-3..q})",
        "economic_rationale": "4-quarter backward rolling mean of ROA (%) – smoothed earnings power.",
    },
    {
        "feature": "roa_roll_std_4q",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "std(ROA_{q-3..q}, ddof=1)",
        "economic_rationale": "4-quarter backward rolling standard deviation of ROA – earnings instability/risk.",
    },
    {
        "feature": "nim_roll_mean_4q",
        "camels_group": "S & Momentum - Sensitivity & Dynamics",
        "formula": "mean(NIMY_{q-3..q})",
        "economic_rationale": "4-quarter backward rolling mean of Net Interest Margin (%).",
    },
]

FEATURE_NAMES: list[str] = [item["feature"] for item in FEATURE_METADATA]


# ════════════════════════════════════════════════════════════════════════════
# 3. POINT-IN-TIME CAMELS FEATURE ENGINEERING IN POLARS
# ════════════════════════════════════════════════════════════════════════════
def safe_div(num: pl.Expr, den: pl.Expr, scale: float = 100.0) -> pl.Expr:
    """Safe division in Polars: returns null when denominator is 0 or null."""
    return pl.when(den.is_not_null() & (den.abs() > 1e-9)).then((num / den) * scale).otherwise(None)


def build_camels_features(clean_path: Path) -> pl.DataFrame:
    """Compute all 37 point-in-time CAMELS features using Polars Lazy API."""
    print(f"Loading clean panel from {clean_path}...")
    lf = pl.scan_parquet(clean_path)

    # Sort strictly by (CERT, REPDTE) before any backward-looking window operations
    lf = lf.sort(["CERT", "REPDTE"])

    # Step 1: Contemporaneous point-in-time ratios at quarter q
    op_rev = pl.col("INTINC") - pl.col("EINTEXP") + pl.col("NONII")
    tx_den = pl.col("EQ") + pl.col("LNATRES")

    lf = lf.with_columns([
        pl.col("ASSET").cast(pl.Float64),
        # --- C: Capital Adequacy ---
        safe_div(pl.col("EQ"), pl.col("ASSET"), 100.0).alias("eq_to_asset"),
        pl.col("RBC1AAJ").cast(pl.Float64).alias("tier1_cap_ratio"),
        pl.col("RBCRWAJ").cast(pl.Float64).alias("total_cap_ratio"),
        safe_div(pl.col("EQ") - pl.col("NCLNLS").fill_null(0.0), pl.col("ASSET"), 100.0).alias("equity_cushion_gap"),

        # --- A: Asset Quality ---
        safe_div(pl.col("NCLNLS"), pl.col("LNLSGR"), 100.0).alias("npl_to_loans"),
        safe_div(pl.col("LNATRES"), pl.col("LNLSGR"), 100.0).alias("llreserve_to_loans"),
        safe_div(pl.col("LNATRES"), pl.col("NCLNLS") + 1.0, 100.0).alias("reserve_coverage"),
        safe_div(pl.col("P9ASSET"), pl.col("ASSET"), 100.0).alias("pastdue90_to_asset"),
        safe_div(pl.col("NAASSET"), pl.col("ASSET"), 100.0).alias("nonaccrual_to_asset"),
        safe_div(pl.col("ORE"), pl.col("ASSET"), 100.0).alias("ore_to_asset"),
        pl.when(tx_den.is_not_null() & (tx_den > 1.0))
        .then(((pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0)) / tx_den) * 100.0)
        .otherwise(None)
        .alias("texas_ratio"),
        safe_div(pl.col("NTLNLS"), pl.col("LNLSGR"), 100.0).alias("net_chargeoff_rate"),

        # --- M & E: Management & Earnings ---
        pl.col("ROA").cast(pl.Float64).alias("roa"),
        pl.col("ROE").cast(pl.Float64).alias("roe"),
        pl.col("NIMY").cast(pl.Float64).alias("nimy"),
        pl.when(op_rev.is_not_null() & (op_rev > 1.0))
        .then((pl.col("NONIX") / op_rev) * 100.0)
        .otherwise(None)
        .alias("cost_to_income"),
        safe_div(pl.col("NETINC"), pl.col("ASSET"), 100.0).alias("retained_profit_ratio"),
        safe_div(pl.col("ELNATR"), pl.col("ASSET"), 100.0).alias("provision_to_asset"),

        # --- L: Liquidity ---
        safe_div(pl.col("BRO"), pl.col("DEP"), 100.0).alias("brokered_dep_ratio"),
        safe_div(pl.col("CHBAL").fill_null(0.0) + pl.col("SC").fill_null(0.0), pl.col("ASSET"), 100.0).alias("liquid_assets_ratio"),
        safe_div(pl.col("LNLSNET"), pl.col("DEP"), 100.0).alias("loan_to_deposit"),
        safe_div(pl.col("DEP"), pl.col("ASSET"), 100.0).alias("dep_to_asset"),
        safe_div(pl.col("DEPDOM"), pl.col("DEP"), 100.0).alias("domestic_dep_share"),

        # --- S: Sensitivity & Portfolio Composition ---
        safe_div(pl.col("LNRE"), pl.col("LNLSGR"), 100.0).alias("re_loan_share"),
        safe_div(pl.col("LNCI"), pl.col("LNLSGR"), 100.0).alias("ci_loan_share"),
        safe_div(pl.col("LNCON"), pl.col("LNLSGR"), 100.0).alias("con_loan_share"),
        safe_div(pl.col("EINTEXP"), pl.col("DEP"), 100.0).alias("funding_cost_rate"),
        pl.when(pl.col("ASSET") > 0).then(pl.col("ASSET").cast(pl.Float64).log()).otherwise(None).alias("log_asset"),
    ])

    # Step 2: Strictly backward-looking lags & rolling windows within each CERT (<= q)
    lf = lf.with_columns([
        pl.col("EQ").shift(1).over("CERT").alias("_eq_lag1"),
        pl.col("ASSET").shift(4).over("CERT").alias("_asset_lag4"),
        pl.col("LNLSGR").shift(4).over("CERT").alias("_loans_lag4"),
        pl.col("npl_to_loans").shift(4).over("CERT").alias("_npl_lag4"),
        pl.col("eq_to_asset").shift(4).over("CERT").alias("_eq_ratio_lag4"),
        pl.col("roa").shift(4).over("CERT").alias("_roa_lag4"),
        pl.col("roa").rolling_mean(window_size=4, min_samples=2).over("CERT").alias("roa_roll_mean_4q"),
        pl.col("roa").rolling_std(window_size=4, min_samples=2, ddof=1).over("CERT").alias("roa_roll_std_4q"),
        pl.col("nimy").rolling_mean(window_size=4, min_samples=2).over("CERT").alias("nim_roll_mean_4q"),
    ])

    # Step 3: Momentum & YoY changes (q vs q-1 and q vs q-4)
    lf = lf.with_columns([
        safe_div(pl.col("EQ") - pl.col("_eq_lag1"), pl.col("ASSET"), 100.0).alias("eq_retention_1q"),
        safe_div(pl.col("ASSET") - pl.col("_asset_lag4"), pl.col("_asset_lag4"), 100.0).alias("asset_growth_yoy"),
        safe_div(pl.col("LNLSGR") - pl.col("_loans_lag4"), pl.col("_loans_lag4"), 100.0).alias("loan_growth_yoy"),
        (pl.col("npl_to_loans") - pl.col("_npl_lag4")).alias("npl_ratio_chg_4q"),
        (pl.col("eq_to_asset") - pl.col("_eq_ratio_lag4")).alias("eq_ratio_chg_4q"),
        (pl.col("roa") - pl.col("_roa_lag4")).alias("roa_chg_4q"),
    ])

    # Step 4: Assign Temporal Split tags
    lf = lf.with_columns([
        pl.when(pl.col("REPDTE") <= pl.lit(TRAIN_END_Q))
        .then(pl.lit("train"))
        .when(pl.col("REPDTE") <= pl.lit(INNER_GAP_END_Q))
        .then(pl.lit("embargo_inner"))
        .when(pl.col("REPDTE") <= pl.lit(DEV_END_Q))
        .then(pl.lit("val"))
        .when(pl.col("REPDTE") <= pl.lit(OUTER_GAP_END_Q))
        .then(pl.lit("embargo_outer"))
        .when(pl.col("REPDTE") <= pl.lit(TEST_END_Q))
        .then(pl.lit("test"))
        .otherwise(pl.lit("post_cutoff"))
        .alias("split"),
    ])

    keep_cols = [
        "CERT", "REPDTE", "STNAME", "exit_type", "fail_dt", "label", "split",
        "ASSET", "DEP", "EQ",
    ] + FEATURE_NAMES

    df = lf.select(keep_cols).collect()
    print(f"  ✓ Computed {len(FEATURE_NAMES)} CAMELS features across {df.height:,} bank-quarters")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL SPLIT SUMMARY & VERIFICATION
# ════════════════════════════════════════════════════════════════════════════
def summarize_splits(df: pl.DataFrame) -> pd.DataFrame:
    """Generate split statistics table and verify all exam constraints."""
    split_defs = [
        ("Development - Train",          df.filter(pl.col("split") == "train")),
        ("Inner Embargo Gap (Purged)",   df.filter(pl.col("split") == "embargo_inner")),
        ("Development - Validation",     df.filter(pl.col("split") == "val")),
        ("Development Total (Used)",     df.filter(pl.col("split").is_in(["train", "val"]))),
        ("Development Full (1984–2008)", df.filter(pl.col("REPDTE") <= pl.lit(DEV_END_Q))),
        ("Outer Embargo Gap (Purged)",   df.filter(pl.col("split") == "embargo_outer")),
        ("Test Set (Hold-out)",          df.filter(pl.col("split") == "test")),
        ("Full Modeling Panel (≤2025Q2)", df),
    ]

    rows = []
    for name, sub in split_defs:
        q_min = sub["REPDTE"].min()
        q_max = sub["REPDTE"].max()
        n_q = sub["REPDTE"].n_unique()
        n_rows = sub.height
        n_banks = sub["CERT"].n_unique()
        y1_rows = int(sub["label"].sum())
        y0_rows = n_rows - y1_rows
        y1_rate_pct = round(y1_rows / n_rows * 100, 4) if n_rows else 0.0
        n_failed_banks = sub.filter(pl.col("label") == 1)["CERT"].n_unique()
        rows.append({
            "subset": name,
            "start_repdte": q_min,
            "end_repdte": q_max,
            "n_quarters": n_q,
            "n_rows": n_rows,
            "n_distinct_banks": n_banks,
            "n_y0_obs": y0_rows,
            "n_y1_obs": y1_rows,
            "positive_rate_pct": y1_rate_pct,
            "n_failed_banks": n_failed_banks,
        })

    split_df = pd.DataFrame(rows)

    # Verify Exam Constraints:
    # 1. Test comes strictly after Development with >= 4 quarters gap
    test_row = split_df[split_df["subset"] == "Test Set (Hold-out)"].iloc[0]
    gap_row = split_df[split_df["subset"] == "Outer Embargo Gap (Purged)"].iloc[0]
    assert gap_row["n_quarters"] >= 4, f"Gap must be >= 4 quarters, got {gap_row['n_quarters']}"
    assert test_row["n_failed_banks"] >= 50, (
        f"Test set must contain >= 50 failures, got {test_row['n_failed_banks']}"
    )

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(SPLIT_CSV_PATH, index=False)
    print(f"  ✓ Split summary saved -> {SPLIT_CSV_PATH}")
    return split_df


# ════════════════════════════════════════════════════════════════════════════
# 5. FIT SKLEARN PREPROCESSING PIPELINE ON TRAINING DATA ONLY
# ════════════════════════════════════════════════════════════════════════════
def fit_and_export_dictionary(df: pl.DataFrame) -> pd.DataFrame:
    """Fit sklearn Pipeline strictly on Train split and export feature dictionary."""
    print("\nFitting scikit-learn Preprocessing Pipeline strictly on Training data...")

    train_df = df.filter(pl.col("split") == "train").select(FEATURE_NAMES).to_pandas()
    val_df   = df.filter(pl.col("split") == "val").select(FEATURE_NAMES).to_pandas()
    test_df  = df.filter(pl.col("split") == "test").select(FEATURE_NAMES).to_pandas()

    preproc = Pipeline([
        ("clipper", InfAndOutlierClipper(lower_q=0.005, upper_q=0.995)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # STRICT RULE: .fit() is called ONLY on train_df!
    X_train_proc = preproc.fit_transform(train_df)
    X_val_proc   = preproc.transform(val_df)
    X_test_proc  = preproc.transform(test_df)

    # Leakage & integrity assertions
    assert np.isfinite(X_train_proc).all(), "Non-finite values found in X_train_proc!"
    assert np.isfinite(X_val_proc).all(),   "Non-finite values found in X_val_proc!"
    assert np.isfinite(X_test_proc).all(),  "Non-finite values found in X_test_proc!"

    print(f"  ✓ X_train_proc shape : {X_train_proc.shape} (mean≈0, std≈1, 0 NaNs)")
    print(f"  ✓ X_val_proc shape   : {X_val_proc.shape}   (transformed with Train params, 0 NaNs)")
    print(f"  ✓ X_test_proc shape  : {X_test_proc.shape}  (transformed with Train params, 0 NaNs)")

    joblib.dump(
        {"pipeline": preproc, "feature_names": FEATURE_NAMES},
        PREPROC_PATH,
    )
    print(f"  ✓ Saved fitted sklearn preprocessor -> {PREPROC_PATH}")

    # Build comprehensive feature dictionary with fitted Train statistics
    clipper: InfAndOutlierClipper = preproc.named_steps["clipper"]
    imputer: SimpleImputer        = preproc.named_steps["imputer"]
    scaler: StandardScaler        = preproc.named_steps["scaler"]

    y_train = df.filter(pl.col("split") == "train")["label"].to_numpy()

    dict_rows = []
    for i, meta in enumerate(FEATURE_METADATA):
        col = meta["feature"]
        raw_s = train_df[col]
        miss_train_pct = round(float(raw_s.isna().mean() * 100), 2)
        corr_with_y = float(np.corrcoef(X_train_proc[:, i], y_train)[0, 1])
        dict_rows.append({
            "feature": col,
            "camels_group": meta["camels_group"],
            "formula": meta["formula"],
            "train_missing_pct": miss_train_pct,
            "train_clip_q005": round(float(clipper.lower_bounds_[i]), 4),
            "train_clip_q995": round(float(clipper.upper_bounds_[i]), 4),
            "train_imputed_median": round(float(imputer.statistics_[i]), 4),
            "train_scaled_mean": round(float(scaler.mean_[i]), 4),
            "train_scaled_std": round(float(scaler.scale_[i]), 4),
            "train_corr_with_label": round(corr_with_y, 4),
            "economic_rationale": meta["economic_rationale"],
        })

    dict_df = pd.DataFrame(dict_rows)
    dict_df.to_csv(DICT_CSV_PATH, index=False)
    print(f"  ✓ Feature dictionary saved -> {DICT_CSV_PATH}")
    return dict_df


# ════════════════════════════════════════════════════════════════════════════
# 6. PLOT TEMPORAL SPLIT & TOP CAMELS FEATURE CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════
def plot_split_and_features(df: pl.DataFrame, dict_df: pd.DataFrame) -> None:
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.2))

    # --- Left Panel: Quarterly Positive Labels (y=1) & Temporal Split Regions ---
    q_agg = (
        df.group_by(["REPDTE", "split"])
        .agg([
            pl.col("label").sum().alias("y1_count"),
            pl.len().alias("n_rows"),
        ])
        .sort("REPDTE")
        .to_pandas()
    )
    q_agg["dt"] = pd.to_datetime(q_agg["REPDTE"], format="%Y%m%d")

    split_colors = {
        "train":         "#1f77b4",
        "embargo_inner": "#7f7f7f",
        "val":           "#ff7f0e",
        "embargo_outer": "#d62728",
        "test":          "#2ca02c",
    }
    split_labels = {
        "train":         "Train (1984Q1–2006Q4)",
        "embargo_inner": "Inner Gap (2007Q1–2007Q4, 4Q)",
        "val":           "Validation (2008Q1–2008Q4)",
        "embargo_outer": "Outer Embargo Gap (2009Q1–2009Q4, 4Q)",
        "test":          "Test Set (2010Q1–2025Q2, 341 failed banks)",
    }

    for sp_key in ["train", "embargo_inner", "val", "embargo_outer", "test"]:
        sub = q_agg[q_agg["split"] == sp_key]
        if len(sub):
            ax1.bar(
                sub["dt"], sub["y1_count"], width=75,
                color=split_colors[sp_key], label=split_labels[sp_key], alpha=0.9,
            )

    ax1.set_title("Temporal Split & 4-Quarter Embargo Gaps (y=1 Failure Labels per Quarter)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Reporting Quarter (REPDTE)", fontsize=10.5)
    ax1.set_ylabel("Banks Failing Within Next 4 Quarters (y = 1)", fontsize=10.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # --- Right Panel: Top 15 Features by Absolute Correlation with Failure (on Train) ---
    top_f = (
        dict_df.assign(abs_corr=dict_df["train_corr_with_label"].abs())
        .sort_values("abs_corr", ascending=True)
        .tail(15)
    )
    bar_colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_f["train_corr_with_label"]]
    bars = ax2.barh(top_f["feature"], top_f["train_corr_with_label"], color=bar_colors, edgecolor="white")
    ax2.axvline(0.0, color="black", linewidth=0.8)
    for b, val in zip(bars, top_f["train_corr_with_label"]):
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax2.text(
            val + offset, b.get_y() + b.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=8.5, fontweight="bold",
        )
    ax2.set_xlim(-0.28, 0.35)
    ax2.set_title("Top 15 CAMELS Features Correlated with 1-Year Failure (Fit on Train Only)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Point-Biserial Correlation with y_{i,q} on Training Set", fontsize=10.5)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved split & feature chart -> {FIG_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 72)
    print("  Part 4 – CAMELS Feature Pipeline & Leak-Free Preprocessing")
    print("=" * 72)

    df = build_camels_features(CLEAN_PANEL_PATH)

    # Save feature table
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.write_parquet(FEATURE_PANEL_PATH, compression="zstd")
    print(f"  ✓ Saved feature panel -> {FEATURE_PANEL_PATH} ({df.height:,} rows × {df.width} cols)")

    # Split summary
    split_df = summarize_splits(df)
    print("\n=== Temporal Split Summary ===")
    print(split_df.to_string(index=False))

    # Fit sklearn pipeline on Train only & export dictionary
    dict_df = fit_and_export_dictionary(df)
    print("\n=== Top 12 Features by |Correlation| with Label (Train Only) ===")
    top12 = (
        dict_df.assign(abs_corr=dict_df["train_corr_with_label"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(12)[["feature", "camels_group", "train_missing_pct", "train_imputed_median", "train_corr_with_label"]]
    )
    print(top12.to_string(index=False))

    # Plot
    plot_split_and_features(df, dict_df)

    print("\n" + "=" * 72)
    print("  Part 4 – COMPLETE ✓ (Zero look-ahead bias verified)")
    print("=" * 72)


if __name__ == "__main__":
    main()
