"""
src/01_storage_benchmark.py – Part 1: Storage Format Benchmark
==============================================================
Compares disk size, write speed, and read speed across six storage layouts:

  1. CSV (raw, no compression)
  2. Parquet + Snappy  (single file)
  3. Parquet + Zstd    (single file)
  4. Parquet + Snappy  + Partitioned by REPDTE
  5. Parquet + Zstd    + Partitioned by REPDTE
  6. Parquet + Snappy  + Partitioned by REPDTE + Sorted by CERT

Measurements
------------
  • Disk size  (MB)
  • Write time (seconds)
  • Full‑scan read time   (read every row & column)
  • Selective read time   (1 quarter, 4 columns – predicate pushdown + column projection)

Outputs
-------
  outputs/tables/part1_storage_benchmark.csv
  outputs/figures/part1_storage_comparison.png

Usage
-----
    python src/01_storage_benchmark.py
"""
from __future__ import annotations

import gc
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data_raw" / "fdic"
BENCH_DIR  = ROOT / "data_raw" / "benchmark"
OUT_TABLE  = ROOT / "outputs" / "tables"  / "part1_storage_benchmark.csv"
OUT_FIGURE = ROOT / "outputs" / "figures" / "part1_storage_comparison.png"

N_READ_ITER = 3          # median of N reads to smooth OS‑cache effects
SELECT_COLS = ["CERT", "ASSET", "DEP", "ROA"]   # columns for selective read


# ── Helpers ─────────────────────────────────────────────────────────────────
def path_size_mb(p: Path) -> float:
    """Total size of a file or directory tree in MB."""
    if p.is_file():
        return p.stat().st_size / 1_048_576
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1_048_576


def median_time(fn, *args, n: int = N_READ_ITER, **kwargs) -> float:
    """Run *fn* n times, return median elapsed seconds."""
    times = []
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def timed(fn, *args, **kwargs):
    """Run once, return (result, elapsed_seconds)."""
    gc.collect()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ── Load source data ───────────────────────────────────────────────────────
def load_financials() -> pd.DataFrame:
    fin_dir = DATA_DIR / "financials"
    if not fin_dir.exists():
        sys.exit(f"ERROR: {fin_dir} not found. Run 01_acquire_and_storage.py first.")
    parquets = sorted(fin_dir.glob("*.parquet"))
    if not parquets:
        sys.exit(f"ERROR: No .parquet files in {fin_dir}.")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    # Ensure REPDTE is string for consistent filtering & partitioning
    df["REPDTE"] = df["REPDTE"].astype(str)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols from {len(parquets)} quarter files")
    return df


# ── Writers ─────────────────────────────────────────────────────────────────
def write_csv(df: pd.DataFrame, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    _, elapsed = timed(df.to_csv, path, index=False)
    return elapsed


def write_parquet_single(df: pd.DataFrame, path: Path, compression: str) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    _, elapsed = timed(df.to_parquet, path, engine="pyarrow",
                       compression=compression, index=False)
    return elapsed


def write_parquet_partitioned(df: pd.DataFrame, root: Path,
                              compression: str,
                              sort_cols: list[str] | None = None) -> float:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    df_out = df.sort_values(sort_cols) if sort_cols else df
    _, elapsed = timed(
        df_out.to_parquet, root, engine="pyarrow",
        compression=compression, partition_cols=["REPDTE"], index=False,
    )
    return elapsed


# ── Readers ─────────────────────────────────────────────────────────────────
def read_full_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_full_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def read_selective_csv(path: Path, quarter: str, cols: list[str]) -> pd.DataFrame:
    use = list(set(cols + ["REPDTE"]))
    df = pd.read_csv(path, usecols=use)
    return df[df["REPDTE"].astype(str) == quarter]


def read_selective_parquet_single(path: Path, quarter: str,
                                  cols: list[str]) -> pd.DataFrame:
    return pd.read_parquet(
        path, engine="pyarrow",
        columns=cols + ["REPDTE"],
        filters=[("REPDTE", "==", quarter)],
    )


def read_selective_parquet_part(root: Path, quarter: str,
                                cols: list[str]) -> pd.DataFrame:
    # For Hive-partitioned datasets (REPDTE=YYYYMMDD), pyarrow infers REPDTE as int64
    # and prunes non-matching partition directories via predicate pushdown.
    return pd.read_parquet(
        root, engine="pyarrow",
        columns=cols,
        filters=[("REPDTE", "==", int(quarter))],
    )


def write_parquet_sorted_rg(df: pd.DataFrame, path: Path,
                            compression: str,
                            sort_cols: list[str],
                            row_group_size: int = 25_000) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.sort_values(sort_cols)
    _, elapsed = timed(
        df_out.to_parquet, path, engine="pyarrow",
        compression=compression, index=False,
        row_group_size=row_group_size,
    )
    return elapsed


# ── Benchmark runner ────────────────────────────────────────────────────────
def run_all(df: pd.DataFrame) -> pd.DataFrame:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    target_q = "20231231" if "20231231" in df["REPDTE"].values else sorted(df["REPDTE"].unique())[-1]
    print(f"  Selective‑read filter: REPDTE == {target_q}, columns = {SELECT_COLS}")

    # Create an unindexed/shuffled copy to represent raw un-sorted ingestion baseline,
    # while sorted variants explicitly sort by (REPDTE, CERT).
    df_unsorted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    results: list[dict] = []

    # -------- 1. CSV (Raw) --------------------------------------------------
    p = BENCH_DIR / "financials.csv"
    wt = write_csv(df_unsorted, p)
    results.append(dict(
        format="CSV (Raw)",
        disk_size_mb=path_size_mb(p),
        write_time_s=wt,
        full_read_time_s=median_time(read_full_csv, p),
        selective_read_time_s=median_time(read_selective_csv, p, target_q, SELECT_COLS),
    ))
    p.unlink(missing_ok=True)
    print("  ✓ CSV (Raw)")

    # -------- 2. Parquet + Snappy (Unsorted Single File) --------------------
    p = BENCH_DIR / "fin_snappy.parquet"
    wt = write_parquet_single(df_unsorted, p, "snappy")
    results.append(dict(
        format="Parquet+Snappy",
        disk_size_mb=path_size_mb(p),
        write_time_s=wt,
        full_read_time_s=median_time(read_full_parquet, p),
        selective_read_time_s=median_time(read_selective_parquet_single,
                                          p, target_q, SELECT_COLS),
    ))
    p.unlink(missing_ok=True)
    print("  ✓ Parquet+Snappy")

    # -------- 3. Parquet + Zstd (Unsorted Single File) ----------------------
    p = BENCH_DIR / "fin_zstd.parquet"
    wt = write_parquet_single(df_unsorted, p, "zstd")
    results.append(dict(
        format="Parquet+Zstd",
        disk_size_mb=path_size_mb(p),
        write_time_s=wt,
        full_read_time_s=median_time(read_full_parquet, p),
        selective_read_time_s=median_time(read_selective_parquet_single,
                                          p, target_q, SELECT_COLS),
    ))
    p.unlink(missing_ok=True)
    print("  ✓ Parquet+Zstd")

    # -------- 4. Parquet + Snappy + Sorted (REPDTE, CERT) -------------------
    p = BENCH_DIR / "fin_snappy_sorted.parquet"
    wt = write_parquet_sorted_rg(df_unsorted, p, "snappy",
                                 sort_cols=["REPDTE", "CERT"], row_group_size=20_000)
    results.append(dict(
        format="PQ+Snappy+Sort(REPDTE,CERT)",
        disk_size_mb=path_size_mb(p),
        write_time_s=wt,
        full_read_time_s=median_time(read_full_parquet, p),
        selective_read_time_s=median_time(read_selective_parquet_single,
                                          p, target_q, SELECT_COLS),
    ))
    p.unlink(missing_ok=True)
    print("  ✓ PQ+Snappy+Sort(REPDTE,CERT)")

    # -------- 5. Parquet + Zstd + Sorted (REPDTE, CERT) ---------------------
    p = BENCH_DIR / "fin_zstd_sorted.parquet"
    wt = write_parquet_sorted_rg(df_unsorted, p, "zstd",
                                 sort_cols=["REPDTE", "CERT"], row_group_size=20_000)
    results.append(dict(
        format="PQ+Zstd+Sort(REPDTE,CERT)",
        disk_size_mb=path_size_mb(p),
        write_time_s=wt,
        full_read_time_s=median_time(read_full_parquet, p),
        selective_read_time_s=median_time(read_selective_parquet_single,
                                          p, target_q, SELECT_COLS),
    ))
    p.unlink(missing_ok=True)
    print("  ✓ PQ+Zstd+Sort(REPDTE,CERT)")

    # -------- 6. Parquet + Snappy + Part(REPDTE) + Sort(CERT) ---------------
    p = BENCH_DIR / "fin_snappy_part_sorted"
    wt = write_parquet_partitioned(df_unsorted, p, "snappy", sort_cols=["REPDTE", "CERT"])
    results.append(dict(
        format="PQ+Snappy+Part+Sort",
        disk_size_mb=path_size_mb(p),
        write_time_s=wt,
        full_read_time_s=median_time(read_full_parquet, p),
        selective_read_time_s=median_time(read_selective_parquet_part,
                                          p, target_q, SELECT_COLS),
    ))
    shutil.rmtree(p, ignore_errors=True)
    print("  ✓ PQ+Snappy+Part+Sort")

    # -------- 7. Parquet + Zstd + Part(REPDTE) + Sort(CERT) -----------------
    p = BENCH_DIR / "fin_zstd_part_sorted"
    wt = write_parquet_partitioned(df_unsorted, p, "zstd", sort_cols=["REPDTE", "CERT"])
    results.append(dict(
        format="PQ+Zstd+Part+Sort",
        disk_size_mb=path_size_mb(p),
        write_time_s=wt,
        full_read_time_s=median_time(read_full_parquet, p),
        selective_read_time_s=median_time(read_selective_parquet_part,
                                          p, target_q, SELECT_COLS),
    ))
    shutil.rmtree(p, ignore_errors=True)
    print("  ✓ PQ+Zstd+Part+Sort")

    # Clean up temporary benchmark directory
    shutil.rmtree(BENCH_DIR, ignore_errors=True)

    return pd.DataFrame(results)


# ── Visualisation ───────────────────────────────────────────────────────────
def plot(res: pd.DataFrame) -> None:
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis", n_colors=len(res))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    metrics = [
        ("disk_size_mb",          "Disk Size (MB)",              axes[0, 0]),
        ("write_time_s",          "Write Time (s)",              axes[0, 1]),
        ("full_read_time_s",      "Full Scan Read (s)",          axes[1, 0]),
        ("selective_read_time_s", "Selective Read (s)",          axes[1, 1]),
    ]

    for col, title, ax in metrics:
        bars = ax.barh(res["format"], res[col], color=palette, edgecolor="white",
                       linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        max_val = res[col].max()
        for bar, val in zip(bars, res[col]):
            label = f"{val:.3f}" if val < 1 else (f"{val:.2f}" if val < 100 else f"{val:.1f}")
            ax.text(bar.get_width() + 0.03 * max_val,
                    bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=9, fontweight="bold")
        ax.set_xlim(0, max_val * 1.30)
        ax.tick_params(axis="y", labelsize=10)

    fig.suptitle("Part 1 – Storage Format Benchmark Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {OUT_FIGURE}")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print("  Part 1 – Storage Format Benchmark")
    print("=" * 64)

    df = load_financials()
    print(f"\nRunning benchmarks (reads repeated {N_READ_ITER}×, median reported)…\n")
    res = run_all(df)

    # Save CSV
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_TABLE, index=False)
    print(f"\nTable saved → {OUT_TABLE}")

    # Pretty‑print
    print("\n" + res.to_string(index=False))

    # Chart
    print()
    plot(res)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
