"""Fetch all data needed to reproduce the paper.

Sources (all free, public):
  - nflverse play-by-play 2002-2024 (one parquet per season)
  - nflverse pbp_participation 2016-2024 (defensive personnel)
  - nflverse weekly_rosters 2002-2024 (player position lookup)
  - nflverse draft_picks (career draft history with gsis_id)
  - Pro Football Reference coaching corpus (1999-2025)
      built locally via scripts/_fetch_coaching.py

Outputs to data/raw/ and data/processed/.

Usage:
    python scripts/00_fetch_data.py [--years 2002 2024]
                                    [--skip-participation]
                                    [--skip-pfr]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

DATASETS = {
    "pbp": {
        "url_pattern": NFLVERSE_BASE + "/pbp/play_by_play_{year}.parquet",
        "out_dir": "pbp",
        "years": list(range(2002, 2025)),
    },
    "participation": {
        "url_pattern": NFLVERSE_BASE + "/pbp_participation/pbp_participation_{year}.parquet",
        "out_dir": "participation",
        "years": list(range(2016, 2025)),
    },
    "rosters": {
        "url_pattern": NFLVERSE_BASE + "/weekly_rosters/roster_weekly_{year}.parquet",
        "out_dir": "rosters",
        "years": list(range(2002, 2025)),
    },
}

DRAFT_URL = NFLVERSE_BASE + "/draft_picks/draft_picks.parquet"


def fetch_one(url: str, dest: Path, force: bool = False) -> bool:
    """Download a single file. Idempotent unless force=True."""
    if dest.exists() and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
    tmp.rename(dest)
    return True


def fetch_dataset(name: str, spec: dict, years: list[int] | None = None) -> int:
    out_dir = RAW / spec["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    target_years = years if years is not None else spec["years"]
    target_years = [y for y in target_years if y in spec["years"]]
    n_fetched = 0
    print(f"\n=== {name} ({len(target_years)} files) ===")
    for y in target_years:
        url = spec["url_pattern"].format(year=y)
        fname = Path(url).name
        dest = out_dir / fname
        try:
            if fetch_one(url, dest):
                n_fetched += 1
        except requests.HTTPError as e:
            print(f"  [skip] {fname}: HTTP {e.response.status_code}")
        except Exception as e:
            print(f"  [error] {fname}: {e}")
    return n_fetched


def fetch_draft():
    out = RAW / "draft" / "draft_picks.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== draft_picks ===")
    try:
        if fetch_one(DRAFT_URL, out):
            print(f"  saved {out}")
    except Exception as e:
        print(f"  [error] draft: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, metavar=("START", "END"),
                   help="Override year range (inclusive both ends)")
    p.add_argument("--skip-participation", action="store_true",
                   help="Skip participation data (needed only for defensive personnel)")
    p.add_argument("--skip-pfr", action="store_true",
                   help="Skip Pro Football Reference coaching scrape")
    args = p.parse_args()

    years = None
    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))

    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"target directory: {RAW}")

    fetch_dataset("nflverse PBP", DATASETS["pbp"], years)
    fetch_dataset("nflverse rosters", DATASETS["rosters"], years)
    if not args.skip_participation:
        fetch_dataset("nflverse participation", DATASETS["participation"], years)
    fetch_draft()

    if not args.skip_pfr:
        print("\n=== Pro Football Reference coaching corpus ===")
        print("Running scripts/_fetch_coaching.py (4 s between requests)...")
        print("This takes ~10-15 minutes on first run; subsequent runs are cached.")
        import subprocess
        rc = subprocess.call([sys.executable, str(REPO / "scripts" / "_fetch_coaching.py")])
        if rc != 0:
            print(f"  PFR scrape exited with code {rc}")
    else:
        print("\nSkipping PFR scrape. Run scripts/_fetch_coaching.py separately when ready.")

    print("\n=== DONE ===")
    print(f"Raw data at: {RAW}")
    print("These raw files are for provenance; the results reproduce from data/frozen (bash reproduce.sh).")


if __name__ == "__main__":
    main()
