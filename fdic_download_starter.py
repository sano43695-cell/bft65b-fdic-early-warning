"""
BFT65B Final Report - Starter script: download FDIC BankFind data via the public API.

What it does
------------
1. /financials   : quarterly Call Report panel, one Parquet file per quarter
                   (resumable: quarters already downloaded and complete are skipped).
2. /failures     : list of failed banks (with COST to the deposit insurance fund).
3. /institutions : static bank information (state, charter class, why reporting ended).
4. Completeness check: rows per quarter == API total, no duplicate (CERT, REPDTE).

No account or API key is needed. Docs: https://api.fdic.gov/banks/docs/

Usage
-----
    pip install requests pandas pyarrow
    # quick test: 4 quarters (~20k rows, ~2-3 minutes)
    python fdic_download_starter.py --start 2023-03-31 --end 2023-12-31
    # full panel, course convention 1984Q1 -> 2026Q2 (~1.68M rows, roughly 1 hour)
    python fdic_download_starter.py

The script is a STARTING POINT. You are expected to understand it, adapt the field
list to your own design, and document your choices in the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000  # maximum rows the API returns per call

# Core field set required by the exam: 3 keys + 30 Call Report fields
# (amounts in thousands of USD; ratios in %).
# Look up every code in the field dictionary before using it:
# https://api.fdic.gov/banks/docs/risview_properties.yaml
FIN_FIELDS = [
    "CERT", "REPDTE", "STNAME",
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",          # balance sheet, liquidity
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",        # loans, reserves
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",     # asset quality
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",      # earnings
    "ROA", "ROE", "NIMY",                                           # earnings ratios
    "RBC1AAJ", "RBCRWAJ",                                           # capital ratios
]
FAIL_FIELDS = ["CERT", "NAME", "CITYST", "FAILDATE", "FAILYR", "COST", "QBFASSET", "QBFDEP", "RESTYPE"]
INST_FIELDS = ["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "ESTYMD", "ENDEFYMD", "CHANGEC1", "INSDATE"]


# ----------------------------------------------------------------------------
# HTTP helper: polite, with retry and exponential back-off
# ----------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BFT65B-course-starter/1.0"})


def get_json(endpoint: str, params: dict, max_retries: int = 6, pause: float = 1.0) -> dict:
    """GET {BASE}/{endpoint} and return the decoded JSON, retrying on errors."""
    url = f"{BASE}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            if r.status_code == 200:
                time.sleep(pause)  # be polite between calls
                return r.json()
            print(f"    HTTP {r.status_code} (attempt {attempt}) - retrying")
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__} (attempt {attempt}) - retrying")
        time.sleep(min(60, 2 ** attempt))  # 2, 4, 8, ... seconds
    raise RuntimeError(f"Giving up on {url} with {params}")


def fetch_all(endpoint: str, fields: list[str], filters: str | None = None,
              sort_by: str = "CERT") -> tuple[pd.DataFrame, int]:
    """Page through an endpoint with limit/offset. Returns (DataFrame, total reported by API)."""
    rows, offset, total = [], 0, None
    while True:
        params = {
            "fields": ",".join(fields),
            "limit": PAGE_SIZE,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": "ASC",
            "format": "json",
        }
        if filters:
            params["filters"] = filters
        js = get_json(endpoint, params)
        total = js["meta"]["total"]
        page = [rec["data"] for rec in js["data"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
    df = pd.DataFrame(rows)
    return df.drop(columns=["ID"], errors="ignore"), total


# ----------------------------------------------------------------------------
# Quarter helpers
# ----------------------------------------------------------------------------
def quarter_ends(start: date, end: date) -> list[str]:
    """All quarter-end dates (YYYYMMDD) between start and end, inclusive."""
    ends = []
    for y in range(start.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if start <= q <= end:
                ends.append(q.strftime("%Y%m%d"))
    return ends


def latest_repdte() -> str:
    js = get_json("financials", {"fields": "REPDTE", "sort_by": "REPDTE",
                                 "sort_order": "DESC", "limit": 1, "format": "json"})
    return js["data"][0]["data"]["REPDTE"]


# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
def download_financials(out: Path, quarters: list[str]) -> None:
    fin_dir = out / "financials"
    fin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest_financials.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for i, q in enumerate(quarters, 1):
        target = fin_dir / f"REPDTE={q}.parquet"
        if target.exists() and manifest.get(q, {}).get("complete"):
            print(f"[{i}/{len(quarters)}] {q}: already complete, skipped")
            continue
        t0 = time.time()
        df, total = fetch_all("financials", FIN_FIELDS, filters=f"REPDTE:{q}")
        if df.empty:
            print(f"[{i}/{len(quarters)}] {q}: no data (quarter not published yet?)")
            continue
        df.to_parquet(target, index=False)  # pyarrow engine, snappy compression
        complete = len(df) == total and not df.duplicated(["CERT", "REPDTE"]).any()
        manifest[q] = {"rows": len(df), "api_total": total, "complete": bool(complete)}
        manifest_path.write_text(json.dumps(manifest, indent=1))  # save after every quarter
        print(f"[{i}/{len(quarters)}] {q}: {len(df):>6,} rows (API total {total:,}) "
              f"{'OK' if complete else 'INCOMPLETE'}  {time.time() - t0:5.1f}s")


def download_small_tables(out: Path) -> None:
    for name, fields, sort_by in (("failures", FAIL_FIELDS, "FAILDATE"),
                                  ("institutions", INST_FIELDS, "CERT")):
        t0 = time.time()
        df, total = fetch_all(name, fields, sort_by=sort_by)
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name:<12}: {len(df):>6,} rows (API total {total:,}) {time.time() - t0:5.1f}s")


# ----------------------------------------------------------------------------
# Completeness check
# ----------------------------------------------------------------------------
def check(out: Path) -> bool:
    manifest = json.loads((out / "manifest_financials.json").read_text())
    panel = pd.read_parquet(out / "financials", columns=["CERT", "REPDTE"])
    n_dup = int(panel.duplicated(["CERT", "REPDTE"]).sum())
    bad = [q for q, m in manifest.items() if not m["complete"]]
    fails = pd.read_parquet(out / "failures.parquet")
    print("\n=== Completeness check ===")
    print(f"quarters downloaded     : {len(manifest)}")
    print(f"total panel rows        : {len(panel):,}")
    print(f"distinct banks (CERT)   : {panel['CERT'].nunique():,}")
    print(f"duplicate CERT-REPDTE   : {n_dup}")
    print(f"incomplete quarters     : {bad if bad else 'none'}")
    print(f"failures in list        : {len(fails):,}")
    return n_dup == 0 and not bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="1984-03-31", help="first quarter-end, YYYY-MM-DD")
    ap.add_argument("--end", default="2026-06-30",
                    help="last quarter-end, YYYY-MM-DD (course convention: 2026-06-30; 'latest' = newest available)")
    ap.add_argument("--out", default="data_raw/fdic", help="output folder (do NOT commit it to Git)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = args.end if args.end != "latest" else pd.to_datetime(latest_repdte()).strftime("%Y-%m-%d")
    quarters = quarter_ends(date.fromisoformat(args.start), date.fromisoformat(end))
    print(f"Downloading {len(quarters)} quarters: {quarters[0]} -> {quarters[-1]} into {out.resolve()}\n")

    download_financials(out, quarters)
    print()
    download_small_tables(out)
    ok = check(out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
