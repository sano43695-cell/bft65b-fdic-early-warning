"""
src/02_data_quality_panel.py – Part 2: Data Quality & Panel
============================================================

Automated pipeline for data quality audit and clean panel construction.

Steps
-----
  1. Load raw data (financials, failures, institutions).
  2. Automated Data Quality Audit:
       • Missing values by column AND by historical era.
       • Anomalous / impossible values.
       • Duplicate (CERT, REPDTE) pairs.
       • Reporting gaps per bank.
  3. Clean Bank-Quarter Panel:
       • Merge three tables via CERT.
       • Classify exits: Failed vs Merged/Other vs Active
         (using failures.FAILDATE + institutions.CHANGEC1).
       • Create label y_{i,q} = 1{bank i fails in (q, q+4 quarters]}.
       • Point-in-time forward-fill (limit 2 quarters, NEVER bfill).
       • Filter to ≤ 2025Q2 (labels after that are incomplete).
  4. Panel summary & failures-timeline chart.

Design Choices (see RULES.md)
------------------------------
  • Forward-fill uses ONLY past data from the SAME bank, max 2 quarters.
  • Backward-fill is NEVER used (would be look-ahead bias).
  • No cross-bank imputation (e.g. industry median fill) at this stage
    — that belongs in the feature pipeline, fit on training data only.
  • Label window (q, q+4Q] ensures at-most-1-year-ahead prediction.
  • Mergers are y=0; only FDIC-confirmed failures are y=1.

Outputs
-------
  outputs/tables/part2_data_quality_report.csv
  outputs/tables/part2_panel_summary.csv
  outputs/figures/part2_failures_timeline.png
  data_processed/clean_panel.parquet

Usage
-----
    python src/02_data_quality_panel.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ───────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data_raw" / "fdic"
DATA_PROC  = ROOT / "data_processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS   = ROOT / "outputs" / "figures"

# ── Constants ───────────────────────────────────────────────────────────
MODELING_CUTOFF = "20250630"      # 2025Q2 – last quarter with complete 4Q label window

# Historical eras for missing-value breakdown
ERAS = {
    "1984–1992": (1984, 1992),    # Pre-Basel / S&L Crisis
    "1993–2000": (1993, 2000),    # Post-Basel I
    "2001–2009": (2001, 2009),    # Pre-GFC + early GFC
    "2010–2019": (2010, 2019),    # Post-GFC recovery
    "2020–2026": (2020, 2026),    # COVID & recent
}

# Columns that must NEVER be forward-filled
NO_FILL = {"CERT", "REPDTE", "STNAME", "exit_type", "fail_dt", "label"}

# CHANGEC1 codes from FDIC data dictionary:
#   200-series = mergers / consolidations (most common: 223 = merger)
#   300-series = charter conversions
#   400-series = other changes
#   500-series = voluntary liquidation
#   600-series = split-off
#   800-series = failures (810/811 = payout, 820 = PA, 850 = receivership)
FAILURE_CODES = {810, 811, 820, 850}
MERGER_CODES  = {211, 213, 215, 217, 221, 222, 223, 224, 225, 230, 235, 240}

# Anomaly detection rules: column → (description, detector_function)
ANOMALY_RULES = {
    "ASSET":   ("≤ 0",               lambda s: s <= 0),
    "DEP":     ("< 0",               lambda s: s < 0),
    "DEPDOM":  ("< 0",               lambda s: s < 0),
    "BRO":     ("< 0",               lambda s: s < 0),
    "CHBAL":   ("< 0",               lambda s: s < 0),
    "LNATRES": ("< 0",               lambda s: s < 0),
    "ROA":     ("|x| > 50",          lambda s: s.abs() > 50),
    "ROE":     ("|x| > 500",         lambda s: s.abs() > 500),
    "NIMY":    ("|x| > 50",          lambda s: s.abs() > 50),
    "RBC1AAJ": ("< −50 or > 200",   lambda s: (s < -50) | (s > 200)),
    "RBCRWAJ": ("< −50 or > 200",   lambda s: (s < -50) | (s > 200)),
    "IDT1CER": ("< −50 or > 200",   lambda s: (s < -50) | (s > 200)),
    "EEFFR":   ("< 0 or > 500",     lambda s: (s < 0) | (s > 500)),
}


# ════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ════════════════════════════════════════════════════════════════════════
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three FDIC tables from raw parquet files."""
    # Financials
    fin_dir = DATA_RAW / "financials"
    parquets = sorted(fin_dir.glob("*.parquet"))
    if not parquets:
        sys.exit(f"ERROR: No parquet files in {fin_dir}. Run 01_acquire_and_storage.py first.")
    fin = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    fin["REPDTE"] = fin["REPDTE"].astype(str).str.strip()
    fin["CERT"]   = pd.to_numeric(fin["CERT"], errors="coerce")
    fin = fin.dropna(subset=["CERT"]).copy()
    fin["CERT"] = fin["CERT"].astype(int)

    # Ensure all financial columns (except keys CERT, REPDTE, STNAME) are numeric float64
    num_candidates = [c for c in fin.columns if c not in ("CERT", "REPDTE", "STNAME")]
    for c in num_candidates:
        fin[c] = pd.to_numeric(fin[c], errors="coerce")

    # Recode disguised structural zero-missingness in Basel capital ratios:
    # Prior to Basel I implementation (1990Q1), FDIC API returns 0.0 instead of NaN
    # for RBCRWAJ (100% of banks in 1984–1989) and RBC1AAJ (~18% Thrifts/S&Ls).
    # Furthermore, any bank with non-zero equity (EQ != 0) or missing EQ cannot
    # genuinely have a 0.000% risk-based capital ratio.
    pre_basel = fin["REPDTE"].astype(int) < 19900331
    eq_nonzero = fin["EQ"].isna() | (fin["EQ"] != 0)
    if "RBCRWAJ" in fin.columns:
        mask_rbcrwaj = (fin["RBCRWAJ"] == 0) & (pre_basel | eq_nonzero)
        fin.loc[mask_rbcrwaj, "RBCRWAJ"] = np.nan
    if "RBC1AAJ" in fin.columns:
        mask_rbc1aaj = (fin["RBC1AAJ"] == 0) & (pre_basel | eq_nonzero)
        fin.loc[mask_rbc1aaj, "RBC1AAJ"] = np.nan

    # Failures
    fail = pd.read_parquet(DATA_RAW / "failures.parquet")
    fail["CERT"] = pd.to_numeric(fail["CERT"], errors="coerce")
    fail = fail.dropna(subset=["CERT"]).copy()
    fail["CERT"] = fail["CERT"].astype(int)
    fail["fail_dt"] = pd.to_datetime(fail["FAILDATE"], errors="coerce")
    n_bad_date = fail["fail_dt"].isna().sum()
    if n_bad_date:
        print(f"  ⚠ {n_bad_date} failures with unparseable FAILDATE – dropped")
        fail = fail.dropna(subset=["fail_dt"]).copy()

    # Institutions
    inst = pd.read_parquet(DATA_RAW / "institutions.parquet")
    inst["CERT"] = pd.to_numeric(inst["CERT"], errors="coerce")
    inst = inst.dropna(subset=["CERT"]).copy()
    inst["CERT"] = inst["CERT"].astype(int)

    print(f"  Financials : {fin.shape[0]:>10,} rows × {fin.shape[1]} cols  "
          f"({fin['REPDTE'].nunique()} quarters, {fin['CERT'].nunique():,} banks)")
    print(f"  Failures   : {fail.shape[0]:>10,} rows")
    print(f"  Institutions: {inst.shape[0]:>10,} rows")
    return fin, fail, inst


# ════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY AUDIT
# ════════════════════════════════════════════════════════════════════════
def audit_quality(fin: pd.DataFrame, fail: pd.DataFrame,
                  inst: pd.DataFrame) -> pd.DataFrame:
    """Column-level quality report across financials, failures, institutions."""
    fin = fin.copy()
    fin["_year"] = fin["REPDTE"].str[:4].astype(int)
    numeric_cols = fin.select_dtypes(include=[np.number]).columns.tolist()
    audit_cols = [c for c in fin.columns if c != "_year"]

    rows: list[dict] = []
    for col in audit_cols:
        r: dict = {
            "column":     col,
            "dtype":      str(fin[col].dtype),
            "total_rows": len(fin),
            "non_null":   int(fin[col].notna().sum()),
            "null_count": int(fin[col].isna().sum()),
            "null_pct":   round(fin[col].isna().mean() * 100, 2),
        }
        # Missing by era
        for era, (y0, y1) in ERAS.items():
            mask = (fin["_year"] >= y0) & (fin["_year"] <= y1)
            sub = fin.loc[mask, col]
            r[f"null%_{era}"] = round(sub.isna().mean() * 100, 2) if len(sub) else np.nan

        # Descriptive stats for numerics
        if col in numeric_cols and fin[col].notna().any():
            r["min"]    = round(float(fin[col].min()), 4)
            r["max"]    = round(float(fin[col].max()), 4)
            r["mean"]   = round(float(fin[col].mean()), 4)
            r["median"] = round(float(fin[col].median()), 4)
        else:
            r["min"] = r["max"] = r["mean"] = r["median"] = ""

        # Anomalies
        if col in ANOMALY_RULES:
            desc, fn = ANOMALY_RULES[col]
            valid = fin[col].dropna()
            anom = int(fn(valid).sum()) if len(valid) else 0
            r["anomaly_rule"]  = desc
            r["anomaly_count"] = anom
        else:
            r["anomaly_rule"]  = ""
            r["anomaly_count"] = 0

        rows.append(r)

    # --- Aggregate audit rows (Financials, Failures, Institutions) ----------
    n_dup = int(fin.duplicated(["CERT", "REPDTE"]).sum())
    rows.append({
        "column": "__DUPLICATE_CERT_REPDTE__",
        "dtype": "audit_financials", "total_rows": len(fin),
        "non_null": len(fin) - n_dup, "null_count": n_dup,
        "null_pct": round(n_dup / len(fin) * 100, 4),
        "anomaly_rule": "Duplicate (CERT, REPDTE)", "anomaly_count": n_dup,
    })

    # Reporting gaps
    fin_s = fin.sort_values(["CERT", "REPDTE"])
    fin_s["_dt"]   = pd.to_datetime(fin_s["REPDTE"], format="%Y%m%d")
    fin_s["_prev"] = fin_s.groupby("CERT")["_dt"].shift(1)
    fin_s["_gap"]  = (fin_s["_dt"] - fin_s["_prev"]).dt.days
    gap_rows = fin_s[fin_s["_gap"] > 120]   # > 1 quarter ≈ 120 days
    banks_w_gap = gap_rows["CERT"].nunique()
    total_gaps  = len(gap_rows)
    rows.append({
        "column": "__REPORTING_GAPS__",
        "dtype": "audit_financials", "total_rows": fin["CERT"].nunique(),
        "non_null": fin["CERT"].nunique() - banks_w_gap,
        "null_count": banks_w_gap,
        "null_pct": round(banks_w_gap / fin["CERT"].nunique() * 100, 2),
        "anomaly_rule": f"Banks with gap > 1Q ({total_gaps:,} instances)",
        "anomaly_count": total_gaps,
    })

    # Audit failures table (missing COST, duplicate CERT)
    fail_cost_null = int(fail["COST"].isna().sum()) if "COST" in fail.columns else 0
    rows.append({
        "column": "__FAILURES_COST_MISSING__",
        "dtype": "audit_failures", "total_rows": len(fail),
        "non_null": len(fail) - fail_cost_null, "null_count": fail_cost_null,
        "null_pct": round(fail_cost_null / len(fail) * 100, 2),
        "anomaly_rule": "Missing DIF resolution COST in failures.parquet",
        "anomaly_count": fail_cost_null,
    })

    # Audit institutions table (duplicate CERT, missing CHANGEC1)
    inst_dup = int(inst.duplicated(["CERT"]).sum())
    rows.append({
        "column": "__INSTITUTIONS_DUP_CERT__",
        "dtype": "audit_institutions", "total_rows": len(inst),
        "non_null": len(inst) - inst_dup, "null_count": inst_dup,
        "null_pct": round(inst_dup / len(inst) * 100, 4),
        "anomaly_rule": "Duplicate CERT in institutions.parquet",
        "anomaly_count": inst_dup,
    })

    report = pd.DataFrame(rows)

    # --- Console summary ----------------------------------------------------
    print("\n" + "=" * 64)
    print("  DATA QUALITY AUDIT")
    print("=" * 64)
    print(f"  Total rows       : {len(fin):,}")
    print(f"  Total banks      : {fin['CERT'].nunique():,}")
    print(f"  Quarters         : {fin['REPDTE'].nunique()}")
    print(f"  Duplicate (C,R)  : {n_dup}")
    print(f"  Banks w/ gaps    : {banks_w_gap:,}  ({total_gaps:,} gap instances)")

    print("\n  ── Key missing-value patterns (% null by era) ──")
    highlight = ["RBC1AAJ", "RBCRWAJ", "IDT1CER", "EEFFR", "NUMEMP",
                 "ASSET", "DEP", "ROA", "NCLNLS", "P3ASSET"]
    for col in highlight:
        if col not in fin.columns:
            continue
        parts = [f"{col:12s}  overall {fin[col].isna().mean()*100:5.1f}%"]
        for era, (y0, y1) in ERAS.items():
            mask = (fin["_year"] >= y0) & (fin["_year"] <= y1)
            pct = fin.loc[mask, col].isna().mean() * 100
            parts.append(f"{era} {pct:5.1f}%")
        print("    " + "  │  ".join(parts))

    print("\n  ── Anomaly counts ──")
    for col, (desc, fn) in ANOMALY_RULES.items():
        if col in fin.columns:
            valid = fin[col].dropna()
            cnt = int(fn(valid).sum()) if len(valid) else 0
            if cnt:
                print(f"    {col:12s}: {cnt:>7,} rows  ({desc})")

    return report


# ════════════════════════════════════════════════════════════════════════
# 3. BUILD CLEAN PANEL
# ════════════════════════════════════════════════════════════════════════
def build_panel(fin: pd.DataFrame, fail: pd.DataFrame,
                inst: pd.DataFrame) -> pd.DataFrame:
    """Merge tables, classify exits, create label, forward-fill, filter."""

    print("\n" + "=" * 64)
    print("  BUILDING CLEAN PANEL")
    print("=" * 64)

    # ------ 3a. Failure lookup (earliest failure per CERT) ------------------
    fail_lookup = (
        fail.groupby("CERT")["fail_dt"]
        .min()
        .reset_index()
        .rename(columns={"fail_dt": "fail_dt"})
    )
    fail_certs = set(fail_lookup["CERT"])

    # ------ 3b. Institution lookup ------------------------------------------
    inst_lk = inst[["CERT", "ACTIVE", "CHANGEC1"]].drop_duplicates("CERT")

    # ------ 3c. Merge -------------------------------------------------------
    panel = fin.copy()
    panel["repdte_dt"] = pd.to_datetime(panel["REPDTE"], format="%Y%m%d")
    panel = panel.merge(fail_lookup, on="CERT", how="left")
    panel = panel.merge(inst_lk,     on="CERT", how="left")

    # ------ 3d. Classify exits ----------------------------------------------
    # Priority: failures table > CHANGEC1 code > ACTIVE flag
    code = panel["CHANGEC1"].fillna(0).astype(int)
    panel["exit_type"] = "Active"

    # Failed: CERT in failures table  OR  CHANGEC1 in failure codes
    panel.loc[panel["CERT"].isin(fail_certs), "exit_type"] = "Failed"
    panel.loc[
        (~panel["CERT"].isin(fail_certs))
        & (code.isin(FAILURE_CODES))
        & (panel["ACTIVE"] == 0),
        "exit_type"
    ] = "Failed"

    # Merged/Other: inactive, not failed
    panel.loc[
        (panel["exit_type"] == "Active") & (panel["ACTIVE"] == 0),
        "exit_type"
    ] = "Merged/Other"

    n_cert = panel["CERT"].nunique()
    for et in ["Failed", "Merged/Other", "Active"]:
        n = panel.loc[panel["exit_type"] == et, "CERT"].nunique()
        print(f"  {et:16s}: {n:>6,} banks  ({n/n_cert*100:.1f}%)")

    # ------ 3e. Create label y_{i,q} ----------------------------------------
    # y = 1 if bank fails in (q, q + 4 quarters]
    # For quarter-end REPDTE, +4 quarters = +1 year exactly (YYYYMMDD + 10000)
    panel["repdte_int"]     = panel["REPDTE"].astype(int)
    panel["window_end_int"] = panel["repdte_int"] + 10000   # +1 year

    # Convert fail_dt to YYYYMMDD integer
    panel["fail_int"] = np.nan
    has_fail = panel["fail_dt"].notna()
    if has_fail.any():
        panel.loc[has_fail, "fail_int"] = (
            panel.loc[has_fail, "fail_dt"].dt.year * 10000
            + panel.loc[has_fail, "fail_dt"].dt.month * 100
            + panel.loc[has_fail, "fail_dt"].dt.day
        )

    panel["label"] = 0
    if has_fail.any():
        label_mask = (
            has_fail
            & (panel["fail_int"] > panel["repdte_int"])
            & (panel["fail_int"] <= panel["window_end_int"])
        )
        panel.loc[label_mask, "label"] = 1

    # Drop rows ON or AFTER failure date (bank no longer operational)
    after_fail = has_fail & (panel["repdte_int"] >= panel["fail_int"])
    n_drop_fail = int(after_fail.sum())
    panel = panel[~after_fail].copy()
    print(f"\n  Dropped {n_drop_fail:,} rows on/after failure date")

    # Filter to modeling cutoff
    n_before = len(panel)
    panel = panel[panel["repdte_int"] <= int(MODELING_CUTOFF)].copy()
    print(f"  Cutoff ≤ {MODELING_CUTOFF}: kept {len(panel):,} rows  "
          f"(dropped {n_before - len(panel):,} post-cutoff)")

    print(f"\n  Label distribution:")
    n0 = (panel["label"] == 0).sum()
    n1 = (panel["label"] == 1).sum()
    print(f"    y=0 : {n0:>10,}")
    print(f"    y=1 : {n1:>10,}   ({n1/(n0+n1)*100:.3f}%)")

    # ------ 3f. Point-in-time anomaly cleaning & forward-fill ---------------
    # Replace impossible negative values in strictly non-negative balance-sheet
    # stock variables with NaN prior to point-in-time forward-fill.
    non_neg_stocks = [
        "DEP", "DEPDOM", "BRO", "CHBAL", "LNATRES",
        "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON",
        "ORE", "NAASSET", "P3ASSET", "P9ASSET", "NCLNLS",
    ]
    n_neg_cleaned = 0
    for c in non_neg_stocks:
        if c in panel.columns:
            neg_mask = panel[c] < 0
            n_neg = int(neg_mask.sum())
            if n_neg:
                panel.loc[neg_mask, c] = np.nan
                n_neg_cleaned += n_neg
    print(f"\n  Impossible negative stock values recoded to NaN: {n_neg_cleaned:,}")

    fill_cols = [
        c for c in panel.columns
        if c not in NO_FILL
        and c not in ("repdte_dt", "repdte_int", "window_end_int",
                      "fail_int", "ACTIVE", "CHANGEC1")
        and panel[c].dtype.kind in ("f", "i")       # numeric only
    ]
    before_null = int(panel[fill_cols].isna().sum().sum())

    panel = panel.sort_values(["CERT", "REPDTE"])
    # Point-in-time forward-fill ONLY across consecutive quarters (gap <= 120 days).
    # If a bank has a reporting gap > 1 quarter, do not ffill across a long gap
    # (or limit=2 quarters within the bank's contiguous reporting segments).
    dt_s = pd.to_datetime(panel["REPDTE"], format="%Y%m%d")
    gap_break = (dt_s.groupby(panel["CERT"]).diff().dt.days > 200).fillna(False)
    seg_id = gap_break.groupby(panel["CERT"]).cumsum()
    panel[fill_cols] = panel.groupby(["CERT", seg_id])[fill_cols].ffill(limit=2)

    after_null = int(panel[fill_cols].isna().sum().sum())
    print(f"\n  Forward-fill (limit=2 quarters, contiguous point-in-time):")
    print(f"    Nulls before : {before_null:>10,}")
    print(f"    Nulls after  : {after_null:>10,}")
    print(f"    Filled       : {before_null - after_null:>10,}")

    # ------ 3g. Clean up helper columns, save --------------------------------
    drop_cols = ["repdte_dt", "repdte_int", "window_end_int",
                 "fail_int", "ACTIVE", "CHANGEC1"]
    panel.drop(columns=[c for c in drop_cols if c in panel.columns],
               inplace=True, errors="ignore")

    return panel


# ════════════════════════════════════════════════════════════════════════
# 4. SUMMARY & VISUALISATION
# ════════════════════════════════════════════════════════════════════════
def panel_summary(panel: pd.DataFrame, fail: pd.DataFrame) -> pd.DataFrame:
    """Yearly counts of reporting banks, failures, and mergers."""
    panel = panel.copy()
    panel["year"] = panel["REPDTE"].str[:4].astype(int)

    # Active (reporting) banks per year
    active = panel.groupby("year")["CERT"].nunique().reset_index(name="n_reporting")

    # Failures per year (from failures table – more complete)
    fail_yr = (
        fail.groupby(fail["fail_dt"].dt.year)["CERT"]
        .nunique()
        .reset_index()
        .rename(columns={"fail_dt": "year", "CERT": "n_failed"})
    )

    # Mergers: banks whose last REPDTE year = this year and exit_type = Merged/Other
    merged = panel[panel["exit_type"] == "Merged/Other"].copy()
    merged_last = merged.groupby("CERT")["year"].max().reset_index()
    merged_yr = (
        merged_last.groupby("year").size().reset_index(name="n_merged")
    )

    # Labeled positives per year
    pos = (
        panel[panel["label"] == 1]
        .groupby("year")["CERT"]
        .nunique()
        .reset_index(name="n_labeled_y1")
    )

    summary = active
    for df_join in [fail_yr, merged_yr, pos]:
        summary = summary.merge(df_join, on="year", how="left")
    summary = summary.fillna(0)
    for c in ["n_failed", "n_merged", "n_labeled_y1"]:
        summary[c] = summary[c].astype(int)

    return summary


def plot_failures(fail: pd.DataFrame) -> None:
    """Bar chart of failures by year with crisis-period shading."""
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    by_year = (
        fail.groupby(fail["fail_dt"].dt.year)
        .size()
        .reset_index(name="n_failures")
        .rename(columns={"fail_dt": "year"})
    )

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(18, 6))

    # Colour-code by crisis period
    colors = []
    for y in by_year["year"]:
        if 1984 <= y <= 1993:
            colors.append("#d62728")     # S&L Crisis
        elif 2008 <= y <= 2013:
            colors.append("#ff7f0e")     # GFC
        elif y >= 2023:
            colors.append("#9467bd")     # Recent
        else:
            colors.append("#1f77b4")     # Normal
    ax.bar(by_year["year"], by_year["n_failures"], color=colors,
           edgecolor="white", linewidth=0.5)

    # Crisis-period shading
    spans = [
        (1984, 1993, "S&L Crisis\n(1984–1993)", "#d62728"),
        (2008, 2013, "GFC\n(2008–2013)",       "#ff7f0e"),
    ]
    for s, e, lbl, c in spans:
        if by_year["year"].min() <= s:
            ax.axvspan(s - 0.5, e + 0.5, alpha=0.07, color=c)
            peak = by_year.loc[
                by_year["year"].between(s, e), "n_failures"
            ].max()
            ax.text((s + e) / 2, peak + 15, lbl,
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=c)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Bank Failures", fontsize=12)
    ax.set_title("US Bank Failures by Year  (FDIC, 1934–2026)",
                 fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.set_xlim(by_year["year"].min() - 1, by_year["year"].max() + 1)
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "part2_failures_timeline.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart → {OUT_FIGS / 'part2_failures_timeline.png'}")


# ════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 64)
    print("  Part 2 – Data Quality & Panel")
    print("=" * 64)

    # 1. Load
    fin, fail, inst = load_raw()

    # 2. Quality audit
    qr = audit_quality(fin, fail, inst)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    qr.to_csv(OUT_TABLES / "part2_data_quality_report.csv", index=False)
    print(f"\n  Report → {OUT_TABLES / 'part2_data_quality_report.csv'}")

    # 3. Build panel
    panel = build_panel(fin, fail, inst)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(DATA_PROC / "clean_panel.parquet", index=False)
    print(f"\n  Clean panel → {DATA_PROC / 'clean_panel.parquet'}")
    print(f"  Shape: {panel.shape[0]:,} rows × {panel.shape[1]} cols")

    # 4. Summary
    print("\n" + "=" * 64)
    print("  PANEL SUMMARY BY YEAR")
    print("=" * 64)
    summary = panel_summary(panel, fail)
    summary.to_csv(OUT_TABLES / "part2_panel_summary.csv", index=False)
    print(f"\n  Summary → {OUT_TABLES / 'part2_panel_summary.csv'}")
    with pd.option_context("display.max_rows", 50, "display.width", 120):
        print(summary.to_string(index=False))

    # 5. Visualisation
    print()
    plot_failures(fail)

    print("\n" + "=" * 64)
    print("  Part 2 – COMPLETE ✓")
    print("=" * 64)


if __name__ == "__main__":
    main()
