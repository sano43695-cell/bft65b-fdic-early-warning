"""
src/01_acquire_and_storage.py – Part 1: Acquisition & Storage
=============================================================
Enhanced version of fdic_download_starter.py.

Downloads FDIC BankFind data via the public API (no key needed):
  1. /financials   – quarterly Call Report panel, one Parquet per quarter (resumable).
  2. /failures     – list of failed banks (with COST to deposit‑insurance fund).
  3. /institutions – static bank information (state, charter class, exit reason).
  4. Completeness check: rows per quarter == API total, no duplicate (CERT, REPDTE).

Field Selection – CAMELS Framework Justification
-------------------------------------------------
The fields are chosen to cover all six CAMELS dimensions used by bank
supervisors to assess institution health:

  C – Capital adequacy
      RBC1AAJ   Tier 1 risk‑based capital ratio
      RBCRWAJ   Total risk‑based capital ratio
      IDT1CER   Tier 1 leverage capital ratio          ← ADDED
      EQ        Total equity capital

  A – Asset quality
      NCLNLS    Noncurrent loans & leases
      P3ASSET   Assets past due 30–89 days
      P9ASSET   Assets past due 90+ days
      NAASSET   Nonaccrual assets
      NTLNLS    Net charge‑offs to loans & leases
      ORE       Other real estate owned
      LNATRES   Allowance for loan & lease losses

  M – Management quality
      EEFFR     Efficiency ratio (nonint. expense / revenue)   ← ADDED
      NONIX     Total noninterest expense

  E – Earnings
      NETINC    Net income
      INTINC    Total interest income
      EINTEXP   Total interest expense
      NONII     Total noninterest income
      ELNATR    Provision for loan & lease losses
      ROA       Return on assets
      ROE       Return on equity
      NIMY      Net interest margin

  L – Liquidity
      CHBAL     Cash & balances due from depository institutions
      SC        Total securities
      DEP       Total deposits
      DEPDOM    Total domestic deposits
      BRO       Brokered deposits
      ASSET     Total assets (size & normalisation denominator)

  S – Sensitivity to market risk (proxied via balance‑sheet composition)
      LNRE      Real estate loans
      LNCI      Commercial & industrial loans
      LNCON     Loans to individuals (consumer)
      LNLSGR    Gross loans & leases
      LNLSNET   Net loans & leases
      SC        Securities (also in L above)

  Size
      NUMEMP    Number of full‑time equivalent employees       ← ADDED

Usage
-----
    pip install requests pandas pyarrow
    # quick test – 4 quarters (~20 k rows, ~2–3 min)
    python src/01_acquire_and_storage.py --start 2023-03-31 --end 2023-12-31
    # full panel 1984 Q1 → 2026 Q2 (~1.68 M rows, ~1 hour)
    python src/01_acquire_and_storage.py
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
PAGE_SIZE = 10_000  # max rows per API call

# ── Financial fields (33 core + 3 supplementary = 36) ──────────────────────
FIN_FIELDS = [
    # Keys
    "CERT", "REPDTE", "STNAME",
    # Balance sheet, size, liquidity  (CAMELS – L, S)
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    # Loans & reserves  (CAMELS – A, S)
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    # Asset quality  (CAMELS – A)
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    # Earnings  (CAMELS – E)
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    # Earnings ratios  (CAMELS – E)
    "ROA", "ROE", "NIMY",
    # Capital ratios  (CAMELS – C)
    "RBC1AAJ", "RBCRWAJ",
    # ── Supplementary CAMELS fields ──
    "IDT1CER",   # Tier 1 leverage capital ratio       (C)
    "EEFFR",     # Efficiency ratio                    (M)
    "NUMEMP",    # Number of full‑time employees       (M / Size)
]

FAIL_FIELDS = [
    "CERT", "NAME", "CITYST", "FAILDATE", "FAILYR",
    "COST", "QBFASSET", "QBFDEP", "RESTYPE",
]

INST_FIELDS = [
    "CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE",
    "ESTYMD", "ENDEFYMD", "CHANGEC1", "INSDATE",
]


# ── HTTP helper: polite, with retry & exponential back‑off ─────────────────
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BFT65B-final-project/1.1"})


def get_json(endpoint: str, params: dict,
             max_retries: int = 6, pause: float = 1.0) -> dict:
    """GET {BASE}/{endpoint} → decoded JSON, with retry."""
    url = f"{BASE}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            if r.status_code == 200:
                time.sleep(pause)
                return r.json()
            print(f"    HTTP {r.status_code} (attempt {attempt}) – retrying")
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__} (attempt {attempt}) – retrying")
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Giving up on {url} with {params}")


def fetch_all(endpoint: str, fields: list[str],
              filters: str | None = None,
              sort_by: str = "CERT") -> tuple[pd.DataFrame, int]:
    """Page through an endpoint with limit/offset.
    Returns (DataFrame, total reported by API).
    """
    rows: list[dict] = []
    offset, total = 0, None
    while True:
        params = {
            "fields": ",".join(fields),
            "limit":  PAGE_SIZE,
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


# ── Quarter helpers ─────────────────────────────────────────────────────────
def quarter_ends(start: date, end: date) -> list[str]:
    """All quarter‑end dates (YYYYMMDD) between start and end, inclusive."""
    ends = []
    for y in range(start.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if start <= q <= end:
                ends.append(q.strftime("%Y%m%d"))
    return ends


def latest_repdte() -> str:
    js = get_json("financials", {
        "fields": "REPDTE", "sort_by": "REPDTE",
        "sort_order": "DESC", "limit": 1, "format": "json",
    })
    return js["data"][0]["data"]["REPDTE"]


# ── Downloads ───────────────────────────────────────────────────────────────
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
        df.to_parquet(target, index=False)
        complete = (len(df) == total
                    and not df.duplicated(["CERT", "REPDTE"]).any())
        manifest[q] = {
            "rows": len(df),
            "api_total": total,
            "complete": bool(complete),
        }
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{i}/{len(quarters)}] {q}: {len(df):>6,} rows "
              f"(API total {total:,}) "
              f"{'OK' if complete else 'INCOMPLETE'}  "
              f"{time.time() - t0:5.1f}s")


def download_small_tables(out: Path, force: bool = False) -> None:
    for name, fields, sort_by in (("failures",     FAIL_FIELDS, "FAILDATE"),
                                  ("institutions", INST_FIELDS, "CERT")):
        target = out / f"{name}.parquet"
        if target.exists() and not force:
            df_existing = pd.read_parquet(target)
            if len(df_existing) > 0:
                print(f"{name:<12}: {len(df_existing):>6,} rows (already downloaded, skipped)")
                continue
        t0 = time.time()
        df, total = fetch_all(name, fields, sort_by=sort_by)
        df.to_parquet(target, index=False)
        print(f"{name:<12}: {len(df):>6,} rows "
              f"(API total {total:,}) {time.time() - t0:5.1f}s")


# ── Completeness check ─────────────────────────────────────────────────────
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
    print(f"duplicate CERT‑REPDTE   : {n_dup}")
    print(f"incomplete quarters     : {bad if bad else 'none'}")
    print(f"failures in list        : {len(fails):,}")
    return n_dup == 0 and not bad


# ── CLI ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--start", default="1984-03-31",
                    help="first quarter‑end, YYYY‑MM‑DD")
    ap.add_argument("--end", default="2026-06-30",
                    help="last quarter‑end, YYYY‑MM‑DD "
                         "(course convention: 2026‑06‑30; 'latest' = newest)")
    ap.add_argument("--out", default="data_raw/fdic",
                    help="output folder (do NOT commit to Git)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end = (args.end if args.end != "latest"
           else pd.to_datetime(latest_repdte()).strftime("%Y-%m-%d"))
    quarters = quarter_ends(date.fromisoformat(args.start),
                            date.fromisoformat(end))
    print(f"Downloading {len(quarters)} quarters: "
          f"{quarters[0]} → {quarters[-1]} into {out.resolve()}\n")

    download_financials(out, quarters)
    print()
    download_small_tables(out)
    ok = check(out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
