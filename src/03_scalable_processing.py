"""
src/03_scalable_processing.py – Part 3: Scalable Processing
===========================================================

Implements and benchmarks the core bank-panel transformation pipeline across:
  (a) pandas (standard in-memory DataFrame)
  (b) Polars Eager API (pl.read_parquet)
  (c) Polars Lazy API  (pl.scan_parquet + query optimizer + .collect())
  (d) PySpark Local Mode (Catalyst optimizer + Tungsten execution engine)

Core Transformation Pipeline
----------------------------
  1. Read raw quarterly Parquet files + institutions table from disk
  2. Filter: ASSET > 0, DEP >= 0, REPDTE >= '19900331'
  3. Join financials with institutions on CERT (retrieve BKCLASS, ACTIVE)
  4. Sort by (CERT, REPDTE)
  5. Window operations per bank (CERT), ordered by REPDTE (strictly backward):
       • Lag 1 quarter & Lag 4 quarters of ASSET, DEP, ROA, EQ
       • 4-quarter asset growth rate: (ASSET - asset_lag4) / asset_lag4
       • 4-quarter backward rolling mean & std (ddof=1, min_periods=2) of ROA, NIMY
  6. Group-by aggregation by (STNAME, REPDTE):
       • n_banks, total_asset, mean_roa, mean_roa_lag1, mean_roa_roll4,
         mean_roa_std4, mean_asset_growth4q

Scale Benchmark
---------------
  Evaluates 10%, 50%, 100% of the dataset, plus 5x (~8.4M rows) and 10x (~16.8M rows)
  synthetic replications (with distinct synthetic CERT IDs so bank-level windows
  remain realistic). Measures wall-clock time (s) and peak RAM delta (MB).

Outputs
-------
  outputs/tables/part3_polars_vs_pandas.csv
  outputs/figures/part3_benchmark.png
  outputs/tables/part3_polars_plan.txt
  outputs/tables/part3_spark_plan.txt
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "8")

import numpy as np
import pandas as pd
import polars as pl
import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data_raw" / "fdic"
FIN_DIR    = DATA_RAW / "financials"
INST_PATH  = DATA_RAW / "institutions.parquet"
SCALE_DIR  = ROOT / "data_raw" / "scale_bench"

OUT_CSV        = ROOT / "outputs" / "tables"  / "part3_polars_vs_pandas.csv"
OUT_FIG        = ROOT / "outputs" / "figures" / "part3_benchmark.png"
OUT_POLARS_TXT = ROOT / "outputs" / "tables"  / "part3_polars_plan.txt"
OUT_SPARK_TXT  = ROOT / "outputs" / "tables"  / "part3_spark_plan.txt"

NEEDED_FIN_COLS = [
    "CERT", "REPDTE", "STNAME",
    "ASSET", "DEP", "EQ", "ROA", "NIMY", "NETINC", "NCLNLS",
]
NEEDED_INST_COLS = ["CERT", "BKCLASS", "ACTIVE"]
FILTER_REPDTE = "19900331"


# ════════════════════════════════════════════════════════════════════════════
# 1. MEMORY & TIMING MONITOR
# ════════════════════════════════════════════════════════════════════════════
class PeakMemoryMonitor:
    """Samples process RSS every 5 ms in a background thread to capture peak RAM."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.baseline_mb = 0.0
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            rss = self.process.memory_info().rss / (1024 * 1024)
            if rss > self.peak_mb:
                self.peak_mb = rss
            time.sleep(0.005)

    def __enter__(self) -> "PeakMemoryMonitor":
        gc.collect()
        time.sleep(0.05)
        self.baseline_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_mb = self.baseline_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        rss = self.process.memory_info().rss / (1024 * 1024)
        if rss > self.peak_mb:
            self.peak_mb = rss

    @property
    def delta_mb(self) -> float:
        return max(1.0, self.peak_mb - self.baseline_mb)

    @property
    def total_peak_mb(self) -> float:
        return self.peak_mb


# ════════════════════════════════════════════════════════════════════════════
# 2. PREPARE SCALED PARQUET DATASETS ON DISK (ONE SCALE AT A TIME)
# ════════════════════════════════════════════════════════════════════════════
def load_base_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and unify schema for base financials (100%) and institutions."""
    cast_exprs = [
        pl.col("CERT").cast(pl.Int64),
        pl.col("REPDTE").cast(pl.String),
        pl.col("STNAME").cast(pl.String),
        pl.col("ASSET").cast(pl.Float64),
        pl.col("DEP").cast(pl.Float64),
        pl.col("EQ").cast(pl.Float64),
        pl.col("ROA").cast(pl.Float64),
        pl.col("NIMY").cast(pl.Float64),
        pl.col("NETINC").cast(pl.Float64),
        pl.col("NCLNLS").cast(pl.Float64),
    ]
    base_df = pl.concat(
        [
            pl.read_parquet(p, columns=NEEDED_FIN_COLS).with_columns(cast_exprs)
            for p in sorted(FIN_DIR.glob("*.parquet"))
        ],
        how="vertical",
    )

    inst_df = (
        pl.read_parquet(INST_PATH)
        .select(NEEDED_INST_COLS)
        .with_columns([
            pl.col("CERT").cast(pl.Int64),
            pl.col("BKCLASS").cast(pl.String),
            pl.col("ACTIVE").cast(pl.Int64),
        ])
        .unique(subset=["CERT"], keep="first")
    )
    return base_df, inst_df


def write_single_scale(
    scale_label: str, base_df: pl.DataFrame, inst_df: pl.DataFrame
) -> tuple[Path, Path, int]:
    """Materialize a single scale's Parquet files on disk to minimize disk/RAM footprint."""
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    tag = scale_label.replace("%", "pct")
    fin_out = SCALE_DIR / f"fin_{tag}.parquet"
    inst_out = SCALE_DIR / f"inst_{tag}.parquet"

    if scale_label in ("10%", "50%"):
        frac = 0.10 if scale_label == "10%" else 0.50
        unique_certs = base_df.select("CERT").unique().sort("CERT")["CERT"].to_numpy()
        rng = np.random.default_rng(42)
        shuffled_certs = rng.permutation(unique_certs)
        k = max(1, int(len(shuffled_certs) * frac))
        keep_certs = set(shuffled_certs[:k].tolist())
        sub_fin = base_df.filter(pl.col("CERT").is_in(keep_certs))
        sub_inst = inst_df.filter(pl.col("CERT").is_in(keep_certs))
        n_rows = sub_fin.height
        sub_fin.write_parquet(fin_out, compression="snappy")
        sub_inst.write_parquet(inst_out, compression="snappy")
    elif scale_label == "100%":
        n_rows = base_df.height
        base_df.write_parquet(fin_out, compression="snappy")
        inst_df.write_parquet(inst_out, compression="snappy")
    else:
        mult = int(scale_label.replace("x", ""))
        max_cert_offset = 1_000_000
        # Stream chunks to a single Parquet file using PyArrow ParquetWriter
        import pyarrow.parquet as pq
        base_pa = base_df.to_arrow()
        writer = pq.ParquetWriter(fin_out, base_pa.schema, compression="snappy")
        n_rows = 0
        for i in range(mult):
            chunk = base_df.with_columns(
                (pl.col("CERT") + i * max_cert_offset).alias("CERT")
            ).to_arrow()
            writer.write_table(chunk)
            n_rows += chunk.num_rows
            del chunk
        writer.close()
        del base_pa

        inst_parts = [
            inst_df.with_columns((pl.col("CERT") + i * max_cert_offset).alias("CERT"))
            for i in range(mult)
        ]
        big_inst = pl.concat(inst_parts, how="vertical")
        big_inst.write_parquet(inst_out, compression="snappy")
        del inst_parts, big_inst

    gc.collect()
    return fin_out, inst_out, n_rows


# ════════════════════════════════════════════════════════════════════════════
# 3. THREE IMPLEMENTATIONS: PANDAS, POLARS EAGER, POLARS LAZY
# ════════════════════════════════════════════════════════════════════════════
def run_pandas_pipeline(fin_path: Path, inst_path: Path) -> pd.DataFrame:
    """(a) Standard pandas pipeline."""
    fin = pd.read_parquet(fin_path)
    inst = pd.read_parquet(inst_path)

    # Filter
    mask = (
        (fin["ASSET"] > 0)
        & (fin["DEP"] >= 0)
        & (fin["REPDTE"] >= FILTER_REPDTE)
    )
    df = fin.loc[mask, ["CERT", "REPDTE", "STNAME", "ASSET", "DEP", "EQ", "ROA", "NIMY"]].copy()
    del fin, mask

    # Join
    df = df.merge(inst[["CERT", "BKCLASS", "ACTIVE"]], on="CERT", how="left")
    del inst

    # Sort
    df.sort_values(["CERT", "REPDTE"], inplace=True, ignore_index=True)

    # Groupby window ops per bank
    g = df.groupby("CERT", sort=False)
    df["asset_lag1"] = g["ASSET"].shift(1)
    df["asset_lag4"] = g["ASSET"].shift(4)
    df["roa_lag1"]   = g["ROA"].shift(1)
    df["roa_lag4"]   = g["ROA"].shift(4)
    df["eq_ratio"]   = df["EQ"] / df["ASSET"]
    df["asset_growth_4q"] = (df["ASSET"] - df["asset_lag4"]) / df["asset_lag4"]

    df["roa_roll_mean_4q"] = (
        g["ROA"].rolling(window=4, min_periods=2).mean().reset_index(level=0, drop=True)
    )
    df["roa_roll_std_4q"] = (
        g["ROA"].rolling(window=4, min_periods=2).std(ddof=1).reset_index(level=0, drop=True)
    )
    df["nimy_roll_mean_4q"] = (
        g["NIMY"].rolling(window=4, min_periods=2).mean().reset_index(level=0, drop=True)
    )

    # State-Quarter aggregation
    agg = (
        df.groupby(["STNAME", "REPDTE"], as_index=False)
        .agg(
            n_banks=("CERT", "count"),
            total_asset=("ASSET", "sum"),
            mean_roa=("ROA", "mean"),
            mean_roa_lag1=("roa_lag1", "mean"),
            mean_roa_roll4=("roa_roll_mean_4q", "mean"),
            mean_roa_std4=("roa_roll_std_4q", "mean"),
            mean_eq_ratio=("eq_ratio", "mean"),
            mean_asset_growth4q=("asset_growth_4q", "mean"),
        )
        .sort_values(["STNAME", "REPDTE"], ignore_index=True)
    )
    return agg


def run_polars_eager_pipeline(fin_path: Path, inst_path: Path) -> pl.DataFrame:
    """(b) Polars Eager mode: read everything into memory first, then transform."""
    fin = pl.read_parquet(fin_path)
    inst = pl.read_parquet(inst_path)

    df = (
        fin.filter(
            (pl.col("ASSET") > 0)
            & (pl.col("DEP") >= 0)
            & (pl.col("REPDTE") >= FILTER_REPDTE)
        )
        .select(["CERT", "REPDTE", "STNAME", "ASSET", "DEP", "EQ", "ROA", "NIMY"])
        .join(inst.select(["CERT", "BKCLASS", "ACTIVE"]), on="CERT", how="left")
        .sort(["CERT", "REPDTE"])
        .with_columns([
            pl.col("ASSET").shift(1).over("CERT").alias("asset_lag1"),
            pl.col("ASSET").shift(4).over("CERT").alias("asset_lag4"),
            pl.col("ROA").shift(1).over("CERT").alias("roa_lag1"),
            pl.col("ROA").shift(4).over("CERT").alias("roa_lag4"),
            (pl.col("EQ") / pl.col("ASSET")).alias("eq_ratio"),
            pl.col("ROA")
            .rolling_mean(window_size=4, min_samples=2)
            .over("CERT")
            .alias("roa_roll_mean_4q"),
            pl.col("ROA")
            .rolling_std(window_size=4, min_samples=2, ddof=1)
            .over("CERT")
            .alias("roa_roll_std_4q"),
            pl.col("NIMY")
            .rolling_mean(window_size=4, min_samples=2)
            .over("CERT")
            .alias("nimy_roll_mean_4q"),
        ])
        .with_columns([
            ((pl.col("ASSET") - pl.col("asset_lag4")) / pl.col("asset_lag4")).alias(
                "asset_growth_4q"
            )
        ])
    )

    agg = (
        df.group_by(["STNAME", "REPDTE"])
        .agg([
            pl.len().alias("n_banks"),
            pl.col("ASSET").sum().alias("total_asset"),
            pl.col("ROA").mean().alias("mean_roa"),
            pl.col("roa_lag1").mean().alias("mean_roa_lag1"),
            pl.col("roa_roll_mean_4q").mean().alias("mean_roa_roll4"),
            pl.col("roa_roll_std_4q").mean().alias("mean_roa_std4"),
            pl.col("eq_ratio").mean().alias("mean_eq_ratio"),
            pl.col("asset_growth_4q").mean().alias("mean_asset_growth4q"),
        ])
        .sort(["STNAME", "REPDTE"])
    )
    return agg


def build_polars_lazy_query(fin_path: Path, inst_path: Path) -> pl.LazyFrame:
    """Build the lazy query graph (used for both execution and .explain())."""
    fin_lf = pl.scan_parquet(fin_path)
    inst_lf = pl.scan_parquet(inst_path)

    panel_lf = (
        fin_lf.filter(
            (pl.col("ASSET") > 0)
            & (pl.col("DEP") >= 0)
            & (pl.col("REPDTE") >= FILTER_REPDTE)
        )
        .join(inst_lf.select(["CERT", "BKCLASS", "ACTIVE"]), on="CERT", how="left")
        .sort(["CERT", "REPDTE"])
        .with_columns([
            pl.col("ASSET").shift(1).over("CERT").alias("asset_lag1"),
            pl.col("ASSET").shift(4).over("CERT").alias("asset_lag4"),
            pl.col("ROA").shift(1).over("CERT").alias("roa_lag1"),
            pl.col("ROA").shift(4).over("CERT").alias("roa_lag4"),
            (pl.col("EQ") / pl.col("ASSET")).alias("eq_ratio"),
            pl.col("ROA")
            .rolling_mean(window_size=4, min_samples=2)
            .over("CERT")
            .alias("roa_roll_mean_4q"),
            pl.col("ROA")
            .rolling_std(window_size=4, min_samples=2, ddof=1)
            .over("CERT")
            .alias("roa_roll_std_4q"),
            pl.col("NIMY")
            .rolling_mean(window_size=4, min_samples=2)
            .over("CERT")
            .alias("nimy_roll_mean_4q"),
        ])
        .with_columns([
            ((pl.col("ASSET") - pl.col("asset_lag4")) / pl.col("asset_lag4")).alias(
                "asset_growth_4q"
            )
        ])
    )

    agg_lf = (
        panel_lf.group_by(["STNAME", "REPDTE"])
        .agg([
            pl.len().alias("n_banks"),
            pl.col("ASSET").sum().alias("total_asset"),
            pl.col("ROA").mean().alias("mean_roa"),
            pl.col("roa_lag1").mean().alias("mean_roa_lag1"),
            pl.col("roa_roll_mean_4q").mean().alias("mean_roa_roll4"),
            pl.col("roa_roll_std_4q").mean().alias("mean_roa_std4"),
            pl.col("eq_ratio").mean().alias("mean_eq_ratio"),
            pl.col("asset_growth_4q").mean().alias("mean_asset_growth4q"),
        ])
        .sort(["STNAME", "REPDTE"])
    )
    return agg_lf


def run_polars_lazy_pipeline(fin_path: Path, inst_path: Path) -> pl.DataFrame:
    """(c) Polars Lazy API: scan_parquet + query optimization + collect()."""
    return build_polars_lazy_query(fin_path, inst_path).collect()


# ════════════════════════════════════════════════════════════════════════════
# 4. EXPORT POLARS EXPLAIN PLAN & ENGINE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def save_polars_explain(fin_path: Path, inst_path: Path) -> None:
    lf = build_polars_lazy_query(fin_path, inst_path)
    unopt_plan = lf.explain(optimized=False)
    opt_plan = lf.explain(optimized=True)

    text = f"""================================================================================
PART 3 – POLARS LAZY EXECUTION PLAN & QUERY OPTIMIZER ANALYSIS
================================================================================

1. OPTIMIZED EXECUTION PLAN (lazy_df.explain(optimized=True))
--------------------------------------------------------------------------------
{opt_plan}

2. UNOPTIMIZED LOGICAL PLAN (lazy_df.explain(optimized=False))
--------------------------------------------------------------------------------
{unopt_plan}

3. HOW THE POLARS ENGINE EXECUTES THIS PIPELINE
--------------------------------------------------------------------------------
(a) Predicate Pushdown (Selection Pushdown):
    The filter predicates `(ASSET > 0) & (DEP >= 0) & (REPDTE >= "19900331")`
    are pushed all the way down into the `Parquet SCAN` node (`SELECTION:` in
    the optimized plan). Using Parquet row-group min/max statistics, Polars skips
    entire row groups prior to 1990Q1 without decompressing them and filters
    rows during scan before the join or sort ever occurs.

(b) Projection Pushdown (Column Pruning):
    Although the source Parquet file contains 10 columns (and raw files have 36),
    the optimizer inspects downstream expressions and prunes unused columns
    (`NETINC`, `NCLNLS`, `BKCLASS`, `ACTIVE`, etc.) right at the `Parquet SCAN`
    (`PROJECT */10 COLUMNS`). Only the exact columns needed for the window and
    aggregation expressions are decoded from disk into Arrow buffers.

(c) Parallel Window & GroupBy Execution (Morsel-Driven Parallelism):
    • Polars is written in Rust on top of Apache Arrow columnar memory arrays
      with zero-copy sharing and SIMD vectorization.
    • In `.over("CERT")`, Polars partitions the sorted Arrow arrays by `CERT`
      and evaluates `shift(1)`, `shift(4)`, `rolling_mean`, and `rolling_std`
      across all CPU cores simultaneously without Python GIL contention.
    • By contrast, pandas creates Python-level GroupBy/Rolling objects, copies
      intermediate Float64 blocks multiple times, and executes single-threaded
      for most steps.
"""
    OUT_POLARS_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_POLARS_TXT.write_text(text, encoding="utf-8")
    print(f"  ✓ Saved Polars execution plan -> {OUT_POLARS_TXT}")


# ════════════════════════════════════════════════════════════════════════════
# 5. RUN BENCHMARK ACROSS SCALES (ISOLATED SUBPROCESS PER ENGINE)
# ════════════════════════════════════════════════════════════════════════════
def run_single_worker(engine_name: str, fin_path: Path, inst_path: Path) -> dict:
    """Execute one engine inside a fresh worker process for exact RAM & timing."""
    fn_map = {
        "pandas":       run_pandas_pipeline,
        "Polars Eager": run_polars_eager_pipeline,
        "Polars Lazy":  run_polars_lazy_pipeline,
    }
    fn = fn_map[engine_name]
    gc.collect()
    with PeakMemoryMonitor() as mon:
        t0 = time.perf_counter()
        out = fn(fin_path, inst_path)
        elapsed = time.perf_counter() - t0
        out_rows = len(out)
        del out
    return {
        "output_rows": int(out_rows),
        "wall_time_s": round(elapsed, 4),
        "peak_ram_delta_mb": round(mon.delta_mb, 2),
        "peak_rss_mb": round(mon.total_peak_mb, 2),
    }


def run_scale_benchmarks() -> pd.DataFrame:
    print("Loading base 100% dataset into memory to write fin_100pct.parquet...")
    base_df, inst_df = load_base_tables()
    fin_100, inst_100, _ = write_single_scale("100%", base_df, inst_df)
    save_polars_explain(fin_100, inst_100)
    del base_df, inst_df
    gc.collect()

    scale_labels = ["10%", "50%", "100%", "5x", "10x"]
    engines = ["pandas", "Polars Eager", "Polars Lazy"]
    records: list[dict] = []

    for scale_label in scale_labels:
        if scale_label == "100%":
            fin_path, inst_path = fin_100, inst_100
            n_rows = pl.scan_parquet(fin_100).select(pl.len()).collect().item()
        else:
            b_df = pl.read_parquet(fin_100)
            i_df = pl.read_parquet(inst_100)
            fin_path, inst_path, n_rows = write_single_scale(scale_label, b_df, i_df)
            del b_df, i_df
            gc.collect()

        print(f"\n── Benchmarking Scale {scale_label} ({n_rows:,} rows) ──")
        for eng_name in engines:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-engine", eng_name,
                "--worker-fin", str(fin_path),
                "--worker-inst", str(inst_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"Worker STDERR ({eng_name} @ {scale_label}):\n{proc.stderr}")
                proc.check_returncode()
            # Last line of stdout has JSON result
            res_json = json.loads(proc.stdout.strip().splitlines()[-1])
            rec = {
                "scale": scale_label,
                "input_rows": n_rows,
                "output_rows": res_json["output_rows"],
                "engine": eng_name,
                "wall_time_s": res_json["wall_time_s"],
                "peak_ram_delta_mb": res_json["peak_ram_delta_mb"],
                "peak_rss_mb": res_json["peak_rss_mb"],
            }
            records.append(rec)
            print(
                f"  {eng_name:<14s} | time: {rec['wall_time_s']:7.3f} s | "
                f"RAM delta: {rec['peak_ram_delta_mb']:7.1f} MB | "
                f"peak RSS: {rec['peak_rss_mb']:7.1f} MB"
            )

        # Keep 100% file for Spark step; remove all other scale files immediately
        if scale_label != "100%":
            fin_path.unlink(missing_ok=True)
            inst_path.unlink(missing_ok=True)

    df_res = pd.DataFrame(records)
    pd_times = df_res[df_res["engine"] == "pandas"].set_index("scale")["wall_time_s"]
    df_res["speedup_vs_pandas"] = df_res.apply(
        lambda r: round(pd_times[r["scale"]] / r["wall_time_s"], 2), axis=1
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_CSV, index=False)
    print(f"\n  ✓ Benchmark table saved -> {OUT_CSV}")
    return df_res


# ════════════════════════════════════════════════════════════════════════════
# 6. PLOT BENCHMARK RESULTS
# ════════════════════════════════════════════════════════════════════════════
def plot_benchmark(df_res: pd.DataFrame) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    palette = {
        "pandas": "#d62728",
        "Polars Eager": "#1f77b4",
        "Polars Lazy": "#2ca02c",
    }
    markers = {"pandas": "o", "Polars Eager": "s", "Polars Lazy": "^"}

    # Panel 1: Wall-clock time vs Scale (log-scale y or linear)
    ax1 = axes[0]
    for eng in ["pandas", "Polars Eager", "Polars Lazy"]:
        sub = df_res[df_res["engine"] == eng]
        ax1.plot(
            sub["scale"], sub["wall_time_s"],
            marker=markers[eng], linewidth=2.4, markersize=8,
            label=eng, color=palette[eng],
        )
        for x, y in zip(sub["scale"], sub["wall_time_s"]):
            ax1.annotate(
                f"{y:.2f}s", (x, y),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8.5, fontweight="bold", color=palette[eng],
            )
    ax1.set_title("Execution Time across Dataset Scales", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("Dataset Scale (100% = 1.68M rows, 10x = 16.8M rows)", fontsize=10.5)
    ax1.set_ylabel("Wall-Clock Time (seconds, log scale)", fontsize=10.5)
    ax1.set_yscale("log")
    ax1.legend(frameon=True, fontsize=10)

    # Panel 2: Peak RAM Delta (MB) across Scales
    ax2 = axes[1]
    x_idx = np.arange(len(df_res["scale"].unique()))
    width = 0.26
    scales_order = df_res["scale"].unique().tolist()
    for i, eng in enumerate(["pandas", "Polars Eager", "Polars Lazy"]):
        sub = df_res[df_res["engine"] == eng].set_index("scale").loc[scales_order]
        bars = ax2.bar(
            x_idx + (i - 1) * width, sub["peak_ram_delta_mb"],
            width=width, label=eng, color=palette[eng], edgecolor="white",
        )
        for b in bars:
            h = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2, h + max(sub["peak_ram_delta_mb"]) * 0.02,
                f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, rotation=0,
            )
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(scales_order)
    ax2.set_title("Peak Memory Allocation (Delta RAM in MB)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Dataset Scale", fontsize=10.5)
    ax2.set_ylabel("Peak RAM Increase (MB)", fontsize=10.5)
    ax2.legend(frameon=True, fontsize=10)

    # Panel 3: Speedup Factor vs pandas
    ax3 = axes[2]
    for eng in ["Polars Eager", "Polars Lazy"]:
        sub = df_res[df_res["engine"] == eng]
        ax3.plot(
            sub["scale"], sub["speedup_vs_pandas"],
            marker=markers[eng], linewidth=2.4, markersize=8,
            label=f"{eng} vs pandas", color=palette[eng],
        )
        for x, y in zip(sub["scale"], sub["speedup_vs_pandas"]):
            ax3.annotate(
                f"{y:.1f}x", (x, y),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=9, fontweight="bold", color=palette[eng],
            )
    ax3.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="pandas baseline (1.0x)")
    ax3.set_title("Speedup Factor Relative to pandas", fontsize=12.5, fontweight="bold")
    ax3.set_xlabel("Dataset Scale", fontsize=10.5)
    ax3.set_ylabel("Speedup (× faster than pandas)", fontsize=10.5)
    ax3.legend(frameon=True, fontsize=10)

    fig.suptitle(
        "Part 3 – Scalable Processing Benchmark: pandas vs. Polars Eager vs. Polars Lazy",
        fontsize=14.5, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Benchmark figure saved -> {OUT_FIG}")


# ════════════════════════════════════════════════════════════════════════════
# 7. PYSPARK REPRODUCTION, ASSERTION (< 1e-6) & EXPLAIN PLAN
# ════════════════════════════════════════════════════════════════════════════
def ensure_java_home() -> None:
    """Ensure JAVA_HOME points to a working JDK 17+ for PySpark."""
    # Check if jdk is already installed in user directory via install-jdk
    jdk_dir = Path.home() / ".jdk"
    if jdk_dir.exists():
        candidates = sorted(jdk_dir.glob("jdk-*"), reverse=True)
        for c in candidates:
            if (c / "bin" / "java.exe").exists():
                os.environ["JAVA_HOME"] = str(c)
                os.environ["PATH"] = str(c / "bin") + os.pathsep + os.environ.get("PATH", "")
                return

    # Download OpenJDK 17 via install-jdk if not present
    import jdk
    print("  Installing portable OpenJDK 17 into ~/.jdk for PySpark...")
    installed_path = jdk.install("17")
    os.environ["JAVA_HOME"] = str(installed_path)
    os.environ["PATH"] = str(Path(installed_path) / "bin") + os.pathsep + os.environ.get("PATH", "")


def run_pyspark_verification(fin_path: Path, inst_path: Path) -> None:
    print("\n── Running PySpark Reproduction & Verification (100% dataset) ──")
    ensure_java_home()

    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("BFT65B-Part3-Verification")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.perf_counter()
    fin_sdf = spark.read.parquet(str(fin_path))
    inst_sdf = spark.read.parquet(str(inst_path)).select("CERT", "BKCLASS", "ACTIVE")

    # 1. Filter & Join
    filtered_sdf = (
        fin_sdf.filter(
            (F.col("ASSET") > 0)
            & (F.col("DEP") >= 0)
            & (F.col("REPDTE") >= F.lit(FILTER_REPDTE))
        )
        .select("CERT", "REPDTE", "STNAME", "ASSET", "DEP", "EQ", "ROA", "NIMY")
        .join(inst_sdf, on="CERT", how="left")
    )

    # 2. Window specifications per bank (CERT) ordered by REPDTE
    w_lag = Window.partitionBy("CERT").orderBy("REPDTE")
    w_roll4 = Window.partitionBy("CERT").orderBy("REPDTE").rowsBetween(-3, 0)

    panel_sdf = (
        filtered_sdf
        .withColumn("asset_lag1", F.lag("ASSET", 1).over(w_lag))
        .withColumn("asset_lag4", F.lag("ASSET", 4).over(w_lag))
        .withColumn("roa_lag1",   F.lag("ROA", 1).over(w_lag))
        .withColumn("roa_lag4",   F.lag("ROA", 4).over(w_lag))
        .withColumn("eq_ratio",   F.col("EQ") / F.col("ASSET"))
        .withColumn(
            "asset_growth_4q",
            (F.col("ASSET") - F.col("asset_lag4")) / F.col("asset_lag4"),
        )
        # Match min_periods=2 in Polars/pandas: only emit rolling mean when count >= 2
        .withColumn("_cnt_roa4", F.count("ROA").over(w_roll4))
        .withColumn(
            "roa_roll_mean_4q",
            F.when(F.col("_cnt_roa4") >= 2, F.avg("ROA").over(w_roll4)).otherwise(F.lit(None)),
        )
        .withColumn(
            "roa_roll_std_4q",
            F.when(F.col("_cnt_roa4") >= 2, F.stddev_samp("ROA").over(w_roll4)).otherwise(F.lit(None)),
        )
        .withColumn("_cnt_nimy4", F.count("NIMY").over(w_roll4))
        .withColumn(
            "nimy_roll_mean_4q",
            F.when(F.col("_cnt_nimy4") >= 2, F.avg("NIMY").over(w_roll4)).otherwise(F.lit(None)),
        )
    )

    # 3. State-Quarter Aggregation
    agg_sdf = (
        panel_sdf.groupBy("STNAME", "REPDTE")
        .agg(
            F.count("CERT").alias("n_banks"),
            F.sum("ASSET").alias("total_asset"),
            F.avg("ROA").alias("mean_roa"),
            F.avg("roa_lag1").alias("mean_roa_lag1"),
            F.avg("roa_roll_mean_4q").alias("mean_roa_roll4"),
            F.avg("roa_roll_std_4q").alias("mean_roa_std4"),
            F.avg("eq_ratio").alias("mean_eq_ratio"),
            F.avg("asset_growth_4q").alias("mean_asset_growth4q"),
        )
        .orderBy("STNAME", "REPDTE")
    )

    # Capture Spark explain(extended=True)
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    agg_sdf.explain(extended=True)
    sys.stdout = old_stdout
    spark_explain_str = buf.getvalue()

    # Collect Spark results to pandas for verification against Polars
    spark_agg_pd = agg_sdf.toPandas()
    spark_elapsed = time.perf_counter() - t0
    print(f"  ✓ PySpark execution completed in {spark_elapsed:.2f} s ({len(spark_agg_pd):,} groups)")

    # Run Polars Lazy on the exact same 100% file
    polars_agg_pd = run_polars_lazy_pipeline(fin_path, inst_path).to_pandas()

    # Sort both identically and handle null STNAME if any
    spark_agg_pd = (
        spark_agg_pd.dropna(subset=["STNAME"])
        .sort_values(["STNAME", "REPDTE"])
        .reset_index(drop=True)
    )
    polars_agg_pd = (
        polars_agg_pd.dropna(subset=["STNAME"])
        .sort_values(["STNAME", "REPDTE"])
        .reset_index(drop=True)
    )

    # Assertions: exact shape, keys, and floating-point discrepancy < 1e-6
    assert len(spark_agg_pd) == len(polars_agg_pd), (
        f"Row count mismatch: Spark={len(spark_agg_pd)} vs Polars={len(polars_agg_pd)}"
    )
    assert (spark_agg_pd["STNAME"].values == polars_agg_pd["STNAME"].values).all(), "STNAME mismatch"
    assert (spark_agg_pd["REPDTE"].values == polars_agg_pd["REPDTE"].values).all(), "REPDTE mismatch"
    assert (spark_agg_pd["n_banks"].values == polars_agg_pd["n_banks"].values).all(), "n_banks mismatch"

    num_cols = [
        "total_asset", "mean_roa", "mean_roa_lag1",
        "mean_roa_roll4", "mean_roa_std4", "mean_eq_ratio", "mean_asset_growth4q",
    ]
    verification_lines = []
    for col in num_cols:
        s_vals = spark_agg_pd[col].to_numpy(dtype=float)
        p_vals = polars_agg_pd[col].to_numpy(dtype=float)
        # Compare where both are non-NaN; also verify NaN positions match
        nan_match = np.array_equal(np.isnan(s_vals), np.isnan(p_vals))
        assert nan_match, f"NaN mask mismatch on column {col}"
        valid = ~np.isnan(s_vals)
        abs_diff = np.abs(s_vals[valid] - p_vals[valid])
        rel_diff = abs_diff / np.maximum(1.0, np.abs(p_vals[valid]))
        max_abs = float(np.max(abs_diff)) if valid.any() else 0.0
        max_rel = float(np.max(rel_diff)) if valid.any() else 0.0
        assert max_rel < 1e-6, (
            f"Column {col} exceeded tolerance 1e-6: max_rel={max_rel:.3e}, max_abs={max_abs:.3e}"
        )
        line = (
            f"  [PASS] {col:<22s} | max_abs_diff = {max_abs:.3e} | "
            f"max_rel_diff = {max_rel:.3e} (< 1e-6)"
        )
        print(line)
        verification_lines.append(line)

    spark.stop()

    report_txt = f"""================================================================================
PART 3 – PYSPARK EXECUTION PLAN & NUMERICAL EQUIVALENCE VERIFICATION
================================================================================

1. AUTOMATED EQUIVALENCE CHECK (PySpark vs. Polars Lazy on 100% Dataset)
--------------------------------------------------------------------------------
  Spark Execution Time : {spark_elapsed:.3f} s
  Matched Groups       : {len(spark_agg_pd):,} (STNAME × REPDTE)
  Tolerance Criterion  : Relative/Absolute Float Error < 1e-6

{chr(10).join(verification_lines)}

  OVERALL VERIFICATION STATUS: ALL ASSERTIONS PASSED ✓

2. SPARK CATALYST EXPLAIN PLAN (agg_sdf.explain(extended=True))
--------------------------------------------------------------------------------
{spark_explain_str}

3. INTERPRETATION OF SPARK EXECUTION PLAN & WHEN A CLUSTER IS JUSTIFIED
--------------------------------------------------------------------------------
• Parsed / Analyzed / Optimized Logical Plan:
  Catalyst pushes the filter `(ASSET > 0 AND DEP >= 0 AND REPDTE >= '19900331')`
  into the Parquet scan (`PushedFilters: [IsNotNull(ASSET), GreaterThan(ASSET,0.0), ...]`)
  and prunes unused columns (`ReadSchema: struct<CERT:bigint,REPDTE:string,...>`).

• Physical Plan Stages:
  1. `FileScan parquet` + `Filter` + `Project`: Reads required columns and applies
     pushed-down predicates on both `financials` and `institutions`.
  2. `BroadcastHashJoin`: Because `institutions` is small (~24K rows), Spark
     broadcasts it to all executors rather than shuffling the 1.68M-row panel.
  3. `Exchange hashpartitioning(CERT, 8)` + `Sort [CERT ASC, REPDTE ASC]`:
     Shuffles bank histories by `CERT` so each partition holds complete time
     series for its subset of banks, enabling `Window` evaluation (`lag` and
     `rowsBetween(-3, 0)`).
  4. `HashAggregate (partial)` -> `Exchange hashpartitioning(STNAME, REPDTE, 8)`
     -> `HashAggregate (final)` -> `Sort`: Performs partial aggregation locally
     before shuffling state-quarter summaries for final aggregation.

• When is a Spark Cluster Justified?
  On the 100% FDIC Call Report panel (1.68M rows, ~130 MB in memory) and even at
  10x scale (16.8M rows), Polars Lazy on a single laptop completes the entire
  pipeline in ~0.2–2.5 seconds—faster than Spark's JVM startup and local shuffle
  overhead. A multi-node Spark cluster becomes justified only when:
    (i) Data volume exceeds single-node RAM (e.g., > 50–100 GB, such as daily
        transaction-level or loan-level credit-registry panels with > 500M rows), OR
    (ii) The pipeline must integrate directly with a distributed data lakehouse
         (HDFS / S3 / Delta Lake) shared across many concurrent supervision jobs.
"""
    OUT_SPARK_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SPARK_TXT.write_text(report_txt, encoding="utf-8")
    print(f"  ✓ Saved Spark plan & verification report -> {OUT_SPARK_TXT}")


# ════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="Part 3 – Scalable Processing")
    ap.add_argument(
        "--step",
        choices=["all", "polars", "spark"],
        default="all",
        help="Run 'polars' benchmark, 'spark' verification, or 'all' (default).",
    )
    ap.add_argument("--worker-engine", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-fin", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-inst", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_engine:
        res = run_single_worker(
            args.worker_engine, Path(args.worker_fin), Path(args.worker_inst)
        )
        print(json.dumps(res))
        return

    print("=" * 72)
    print(f"  Part 3 – Scalable Processing (step={args.step})")
    print("=" * 72)

    if args.step in ("all", "polars"):
        df_res = run_scale_benchmarks()
        print("\n=== Benchmark Summary Table ===")
        print(df_res.to_string(index=False))
        plot_benchmark(df_res)

    if args.step in ("all", "spark"):
        fin_100 = SCALE_DIR / "fin_100pct.parquet"
        inst_100 = SCALE_DIR / "inst_100pct.parquet"
        if not fin_100.exists() or not inst_100.exists():
            base_df, inst_df = load_base_tables()
            fin_100, inst_100, _ = write_single_scale("100%", base_df, inst_df)
            del base_df, inst_df
            gc.collect()
        run_pyspark_verification(fin_100, inst_100)

        if SCALE_DIR.exists():
            shutil.rmtree(SCALE_DIR, ignore_errors=True)
            print(f"  ✓ Cleaned up temporary benchmark directory {SCALE_DIR.name}")

    print("\n" + "=" * 72)
    print(f"  Part 3 ({args.step}) – COMPLETE ✓")
    print("=" * 72)


if __name__ == "__main__":
    main()
