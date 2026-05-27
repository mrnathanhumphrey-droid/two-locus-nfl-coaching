"""Phase 4: fetch PFR per-coach pages for HC + OC + DC scope.

Reads slug list from data/processed/coaching/coach_slugs.parquet,
filtered to fetch_lineage=True (~411 slugs).

Uses cf_clearance cookies + UA from data/raw/pfr/_session.py
(handed off from user's browser; same that worked for the 829-page run).

Throttle: 4s between requests. Cache HTML at
data/raw/pfr/raw_coaches/{slug}.

If a 403 (Cloudflare challenge re-served) occurs, halt and report to user
to refresh cookies.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
from curl_cffi import requests as cc

# Pull session config from PFR side dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "raw" / "pfr".resolve()))
from _session import USER_AGENT, CF_CLEARANCE_1, CF_CLEARANCE_2

THROTTLE = 4.0
RAW_COACHES = Path(__file__).resolve().parent.parent / "data" / "raw" / "pfr" / "raw_coaches"
RAW_COACHES.mkdir(parents=True, exist_ok=True)

SLUGS_FP = Path(__file__).resolve().parent.parent / "data" / "processed" / "coach_slugs.parquet"


def make_session() -> cc.Session:
    s = cc.Session(impersonate="chrome120")
    s.headers["User-Agent"] = USER_AGENT
    s.headers["Accept-Language"] = "en-US,en;q=0.9"
    s.cookies.set("cf_clearance", CF_CLEARANCE_1, domain=".pro-football-reference.com")
    s.cookies.set("cf_clearance", CF_CLEARANCE_2, domain="www.pro-football-reference.com")
    return s


def fetch_slug(sess, slug: str) -> tuple[bool, str]:
    fp = RAW_COACHES / slug
    if fp.exists() and fp.stat().st_size > 1024:
        return True, "cache"
    if fp.exists() and fp.stat().st_size == 0:
        return False, "cache-404"
    url = f"https://www.pro-football-reference.com/coaches/{slug}"
    for attempt in range(3):
        try:
            r = sess.get(url, timeout=45)
        except Exception as e:
            return False, f"net-err: {type(e).__name__}"
        if r.status_code == 200:
            fp.write_text(r.text, encoding="utf-8")
            time.sleep(THROTTLE)
            return True, "200"
        if r.status_code == 404:
            fp.write_text("", encoding="utf-8")
            time.sleep(THROTTLE)
            return False, "404"
        if r.status_code == 429:
            print(f"    429 on {slug}, sleeping 60s")
            time.sleep(60)
            continue
        if r.status_code == 403:
            return False, "403-CHALLENGE-COOKIES-EXPIRED"
        return False, f"http-{r.status_code}"
    return False, "429-exhausted"


def main():
    df = pd.read_parquet(SLUGS_FP)
    fetch = df[df["fetch_lineage"]].copy().sort_values("sr_slug")
    print(f"total slugs to fetch: {len(fetch)} (HC + OC + DC scope)")

    sess = make_session()
    ok = 0
    fail = 0
    cache_hit = 0
    fetched = 0
    failures = []
    t0 = time.time()
    for i, row in enumerate(fetch.itertuples(index=False), 1):
        slug = row.sr_slug
        success, status = fetch_slug(sess, slug)
        if status == "cache":
            cache_hit += 1
            ok += 1
        elif status == "200":
            fetched += 1
            ok += 1
        else:
            fail += 1
            failures.append((slug, row.coach_name, status))
            # If cookies expired, halt early — don't burn through the list with 403s
            if status == "403-CHALLENGE-COOKIES-EXPIRED":
                print(f"\n*** COOKIES EXPIRED at slug={slug}. HALTING ***")
                print(f"    Need fresh cf_clearance from user.")
                break
        if i % 25 == 0 or i == len(fetch):
            rate = i / max(time.time() - t0, 1)
            eta_min = (len(fetch) - i) / max(rate, 0.001) / 60
            print(f"  [{i:>4}/{len(fetch)}] {slug:18s} {row.coach_name[:24]:24s} "
                  f"{status:18s} ETA={eta_min:.1f}min "
                  f"ok={ok} fail={fail}")

    elapsed = (time.time() - t0) / 60
    print()
    print(f"=== SUMMARY ===")
    print(f"attempted: {ok + fail}")
    print(f"  cache hits: {cache_hit}")
    print(f"  fresh fetches: {fetched}")
    print(f"  failures: {fail}")
    print(f"wall-clock: {elapsed:.1f} min")
    if failures:
        print(f"\nfailure log:")
        for slug, name, status in failures[:30]:
            print(f"  {slug:18s} {name:30s} {status}")


if __name__ == "__main__":
    main()
