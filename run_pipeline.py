"""
run_pipeline.py – Master End-to-End Pipeline Runner (BFT65B Final Report)
=========================================================================

Executes the entire Bank Failure Early-Warning Data Engineering Pipeline
from Part 1 to Part 5 with a single command, verifying all output tables
and figures required by the examination rubric.

Pipeline Stages
---------------
  • Part 1a : src/01_acquire_and_storage.py   (FDIC API acquisition & completeness check)
  • Part 1b : src/01_storage_benchmark.py     (CSV vs. Parquet Snappy/Zstd/Sorted/Partitioned)
  • Part 2  : src/02_data_quality_panel.py    (Automated quality audit & point-in-time panel)
  • Part 3  : src/03_scalable_processing.py   (Polars vs. pandas benchmark & PySpark verification)
  • Part 4  : src/04_feature_pipeline.py      (37 CAMELS point-in-time features & leak-free Pipeline)
  • Part 5  : src/05_modelling_validation.py  (Expanding-Window CV, MLlib, cost p*, single Test eval)

Usage
-----
    # Full end-to-end pipeline (resumes automatically if raw data already downloaded)
    python run_pipeline.py

    # Quick smoke test on 4 quarters of 2023 for acquisition step
    python run_pipeline.py --quick-acquire

    # Verify all generated tables and figures without re-running heavy steps
    python run_pipeline.py --verify-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

EXPECTED_ARTIFACTS = [
    # Tables
    ROOT / "outputs" / "tables" / "part1_storage_benchmark.csv",
    ROOT / "outputs" / "tables" / "part2_data_quality_report.csv",
    ROOT / "outputs" / "tables" / "part2_panel_summary.csv",
    ROOT / "outputs" / "tables" / "part3_polars_vs_pandas.csv",
    ROOT / "outputs" / "tables" / "part3_polars_plan.txt",
    ROOT / "outputs" / "tables" / "part3_spark_plan.txt",
    ROOT / "outputs" / "tables" / "part4_split_summary.csv",
    ROOT / "outputs" / "tables" / "part4_feature_dictionary.csv",
    ROOT / "outputs" / "tables" / "part5_cv_results.csv",
    ROOT / "outputs" / "tables" / "part5_model_comparison.csv",
    # Figures
    ROOT / "outputs" / "figures" / "part1_storage_comparison.png",
    ROOT / "outputs" / "figures" / "part2_failures_timeline.png",
    ROOT / "outputs" / "figures" / "part3_benchmark.png",
    ROOT / "outputs" / "figures" / "part4_split_and_features.png",
    ROOT / "outputs" / "figures" / "part5_pr_roc_curves.png",
    ROOT / "outputs" / "figures" / "part5_cost_threshold.png",
    ROOT / "outputs" / "figures" / "part5_feature_importance.png",
]


def run_step(title: str, cmd: list[str]) -> float:
    print("\n" + "━" * 76)
    print(f"▶ STAGE: {title}")
    print(f"  Command: {' '.join(cmd)}")
    print("━" * 76)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.exit(f"\n✖ ERROR: Stage '{title}' failed with exit code {proc.returncode}.")
    print(f"✓ Completed '{title}' in {elapsed:.1f}s")
    return elapsed


def verify_artifacts() -> bool:
    print("\n" + "=" * 76)
    print("  REPRODUCIBILITY & ARTIFACT VERIFICATION CHECK")
    print("=" * 76)
    all_ok = True
    for path in EXPECTED_ARTIFACTS:
        rel = path.relative_to(ROOT).as_posix()
        if path.exists() and path.stat().st_size > 0:
            kb = path.stat().st_size / 1024.0
            print(f"  [OK]   {rel:<48s} ({kb:>8.1f} KB)")
        else:
            print(f"  [MISS] {rel:<48s} (NOT FOUND OR EMPTY)")
            all_ok = False
    print("=" * 76)
    if all_ok:
        print("  ALL 17 REQUIRED TABLES, PLANS & FIGURES VERIFIED SUCCESSFULLY ✓")
    else:
        print("  WARNING: One or more artifacts are missing!")
    print("=" * 76)
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--quick-acquire",
        action="store_true",
        help="Pass --start 2023-03-31 --end 2023-12-31 to 01_acquire_and_storage.py",
    )
    ap.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip Part 1 & Part 3 synthetic scaling benchmarks if already generated",
    )
    ap.add_argument(
        "--skip-spark",
        action="store_true",
        help="Skip local PySpark JVM steps if running in a restricted socket environment",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that all output tables, plans, and figures exist and are non-empty",
    )
    args = ap.parse_args()

    if args.verify_only:
        ok = verify_artifacts()
        sys.exit(0 if ok else 1)

    t_start = time.perf_counter()
    py = sys.executable

    # 1. Part 1a: Data Acquisition & Completeness Verification
    acq_cmd = [py, "src/01_acquire_and_storage.py"]
    if args.quick_acquire:
        acq_cmd += ["--start", "2023-03-31", "--end", "2023-12-31"]
    run_step("Part 1a – FDIC Data Acquisition & Completeness Check", acq_cmd)

    # 2. Part 1b: Storage Layout Benchmark
    if not args.skip_benchmark:
        run_step("Part 1b – Storage Format & Partition/Sort Benchmark", [py, "src/01_storage_benchmark.py"])

    # 3. Part 2: Data Quality Audit & Point-in-Time Clean Panel
    run_step("Part 2 – Automated Data Quality Audit & Clean Panel Construction", [py, "src/02_data_quality_panel.py"])

    # 4. Part 3: Scalable Processing (Polars vs. pandas & PySpark Verification)
    if not args.skip_benchmark:
        step_arg = "polars" if args.skip_spark else "all"
        run_step(
            f"Part 3 – Scalable Processing Benchmark & Explain Plans (step={step_arg})",
            [py, "src/03_scalable_processing.py", "--step", step_arg],
        )

    # 5. Part 4: Point-in-Time CAMELS Feature Pipeline & Leak-Free Preprocessor
    run_step("Part 4 – CAMELS Feature Pipeline & Temporal Split", [py, "src/04_feature_pipeline.py"])

    # 6. Part 5: Modelling, Validation, Cost Thresholding & Single Test Evaluation
    m_step = "sklearn" if args.skip_spark else "all"
    run_step(
        f"Part 5 – Modelling, Expanding-Window CV & Single Test Evaluation (step={m_step})",
        [py, "src/05_modelling_validation.py", "--step", m_step],
    )

    ok = verify_artifacts()
    total_min = (time.perf_counter() - t_start) / 60.0
    print(f"\nTotal End-to-End Pipeline Elapsed Time: {total_min:.2f} minutes.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
