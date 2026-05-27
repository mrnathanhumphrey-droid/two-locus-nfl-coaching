"""PFR coaching history scrape — 32 active franchises x 1999-2024.

Uses cf_clearance cookies handed off from user's browser session.
PFR uses single franchise abbr across relocations; team name in page header
reflects the era.

Output:
  data/raw/pfr/coaching_history.csv         (wide, one row per team-season)
  data/raw/pfr/coaching_history_long.csv    (long, one row per team-season-role-coach)
  data/raw/pfr/coaching_data_quality.md     (coverage report)
  data/raw/pfr/raw/{abbr}_{year}.html       (HTML cache, idempotent re-runs)
"""
from __future__ import annotations
import sys
import re
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cc

sys.path.insert(0, str(Path(__file__).parent))
from _session import USER_AGENT, CF_CLEARANCE_1, CF_CLEARANCE_2

THROTTLE = 4.0
MAX_RETRY_ON_429 = 3
ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "pfr"
RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# 32 active franchises, single PFR abbr each (era-spanning)
TEAMS = [
    ("buf", "Buffalo Bills"),       ("mia", "Miami Dolphins"),
    ("nwe", "New England Patriots"),("nyj", "New York Jets"),
    ("rav", "Baltimore Ravens"),    ("cin", "Cincinnati Bengals"),
    ("cle", "Cleveland Browns"),    ("pit", "Pittsburgh Steelers"),
    ("htx", "Houston Texans"),      ("clt", "Indianapolis Colts"),
    ("jax", "Jacksonville Jaguars"),("oti", "Tennessee Titans"),
    ("den", "Denver Broncos"),      ("kan", "Kansas City Chiefs"),
    ("sdg", "Los Angeles Chargers"),("rai", "Las Vegas Raiders"),
    ("dal", "Dallas Cowboys"),      ("nyg", "New York Giants"),
    ("phi", "Philadelphia Eagles"), ("was", "Washington Commanders"),
    ("chi", "Chicago Bears"),       ("det", "Detroit Lions"),
    ("gnb", "Green Bay Packers"),   ("min", "Minnesota Vikings"),
    ("atl", "Atlanta Falcons"),     ("car", "Carolina Panthers"),
    ("nor", "New Orleans Saints"),  ("tam", "Tampa Bay Buccaneers"),
    ("crd", "Arizona Cardinals"),   ("ram", "Los Angeles Rams"),
    ("sea", "Seattle Seahawks"),    ("sfo", "San Francisco 49ers"),
]
YEARS = list(range(1999, 2026))   # 1999..2025
HTX_FIRST_YEAR = 2002


def make_session() -> cc.Session:
    s = cc.Session(impersonate="chrome120")
    s.headers["User-Agent"] = USER_AGENT
    s.headers["Accept-Language"] = "en-US,en;q=0.9"
    s.cookies.set("cf_clearance", CF_CLEARANCE_1, domain=".pro-football-reference.com")
    s.cookies.set("cf_clearance", CF_CLEARANCE_2, domain="www.pro-football-reference.com")
    return s


def fetch(sess: cc.Session, abbr: str, year: int) -> tuple[str | None, str]:
    """Returns (html_or_None, status_string). Cached on disk."""
    fp = RAW / f"{abbr}_{year}.html"
    if fp.exists():
        size = fp.stat().st_size
        if size > 1024:
            return fp.read_text(encoding="utf-8"), "cache-hit"
        elif size == 0:
            return None, "cache-404"

    url = f"https://www.pro-football-reference.com/teams/{abbr}/{year}.htm"
    for attempt in range(MAX_RETRY_ON_429):
        try:
            r = sess.get(url, timeout=45)
        except Exception as e:
            return None, f"net-err: {type(e).__name__}: {str(e)[:60]}"
        if r.status_code == 200:
            fp.write_text(r.text, encoding="utf-8")
            time.sleep(THROTTLE)
            return r.text, "200"
        if r.status_code == 404:
            fp.write_text("", encoding="utf-8")  # negative cache
            time.sleep(THROTTLE)
            return None, "404"
        if r.status_code == 429:
            print(f"    429 on {abbr}/{year}, sleeping 60s")
            time.sleep(60)
            continue
        if r.status_code == 403:
            return None, "403-CHALLENGE-BACK"
        return None, f"http-{r.status_code}"
    return None, "429-exhausted"


# Coaching role label -> output column name
ROLE_LABELS = {
    "coach": "head_coach",
    "head coach": "head_coach",
    "offensive coordinator": "offensive_coordinator",
    "defensive coordinator": "defensive_coordinator",
    "special teams coach": "special_teams_coordinator",
    "special teams coordinator": "special_teams_coordinator",
    "assistant head coach": "assistant_head_coach",
}

RECORD_RE = re.compile(r"\((\d+)-(\d+)(?:-(\d+))?\)")


def parse_coach_text(value: str) -> list[dict]:
    """Parse a coach-line value, returning list of dicts:
       [{name, wins, losses, ties, games}].
    Handles 'Name (W-L)' and 'Name1 (W-L) and Name2 (W-L)'.
    """
    if not value:
        return []
    parts = re.split(r"\s+and\s+", value, flags=re.IGNORECASE)
    out = []
    for p in parts:
        rec = RECORD_RE.search(p)
        if rec:
            name = p[:rec.start()].strip().rstrip(",")
            w, l = int(rec.group(1)), int(rec.group(2))
            t = int(rec.group(3)) if rec.group(3) else 0
            out.append({"name": name, "wins": w, "losses": l, "ties": t,
                        "games": w + l + t})
        else:
            # no record — likely OC/DC/ST (no won-loss attached)
            out.append({"name": p.strip().rstrip(","),
                        "wins": None, "losses": None, "ties": None, "games": None})
    return out


def parse_meta(html: str) -> dict:
    """Extract coaching info from PFR team-season page meta div.
    Returns dict keyed by role -> raw value string + parsed coaches list.
    """
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("div", id="meta")
    if meta is None:
        # fallback: search whole page
        meta = soup
    out = {"raw": {}, "parsed": {}, "team_full_name_h1": ""}

    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(" ", strip=True)
        out["team_full_name_h1"] = re.sub(r"\s+", " ", h1_text)

    for p in meta.find_all("p"):
        strong = p.find("strong")
        if strong is None:
            continue
        label = strong.get_text(strip=True).rstrip(":").lower()
        if label not in ROLE_LABELS:
            continue
        # strip the strong tag, take rest of <p> text
        strong.extract()
        value = p.get_text(" ", strip=True)
        value = re.sub(r"^[:\s]+", "", value)
        role_col = ROLE_LABELS[label]
        # If label appears twice (rare), keep first
        if role_col in out["raw"]:
            continue
        out["raw"][role_col] = value
        out["parsed"][role_col] = parse_coach_text(value)
    return out


def main():
    sess = make_session()
    tasks = []
    for abbr, full in TEAMS:
        for y in YEARS:
            if abbr == "htx" and y < HTX_FIRST_YEAR:
                continue
            tasks.append((abbr, full, y))
    print(f"total team-seasons to fetch: {len(tasks)}")

    wide_rows = []
    long_rows = []
    failures = []
    fetched_count = 0
    cache_count = 0
    parsed_ok = 0
    parse_failed = 0
    t0 = time.time()

    for i, (abbr, full, year) in enumerate(tasks, 1):
        html, status = fetch(sess, abbr, year)
        if status == "cache-hit":
            cache_count += 1
        elif status == "200":
            fetched_count += 1
        else:
            failures.append((abbr, year, status))
            wide_rows.append({
                "season": year, "team": abbr, "team_full_name": full,
                "head_coach": None, "head_coach_record": None,
                "head_coach_games_split": None,
                "offensive_coordinator": None, "defensive_coordinator": None,
                "special_teams_coordinator": None, "assistant_head_coach": None,
                "notes": f"FETCH_FAIL: {status}",
                "source_url": f"https://www.pro-football-reference.com/teams/{abbr}/{year}.htm",
            })
            if i % 25 == 0:
                rate = i / max(time.time() - t0, 1)
                eta_min = (len(tasks) - i) / max(rate, 0.001) / 60
                print(f"  [{i:>4}/{len(tasks)}] {abbr}/{year} {status:18s} ETA={eta_min:.1f}min")
            continue

        if not html:
            failures.append((abbr, year, status))
            continue

        try:
            parsed = parse_meta(html)
            parsed_ok += 1
        except Exception as e:
            parse_failed += 1
            failures.append((abbr, year, f"parse-err: {type(e).__name__}: {str(e)[:60]}"))
            continue

        # Wide row
        head_coaches = parsed["parsed"].get("head_coach", [])
        head_coach_names = ", ".join(c["name"] for c in head_coaches) if head_coaches else None
        head_coach_record = parsed["raw"].get("head_coach")
        head_coach_split = (head_coach_record
                            if (head_coach_record and " and " in head_coach_record.lower())
                            else None)

        wide_rows.append({
            "season": year,
            "team": abbr,
            "team_full_name": full,
            "head_coach": head_coach_names,
            "head_coach_record": head_coach_record,
            "head_coach_games_split": head_coach_split,
            "offensive_coordinator": parsed["raw"].get("offensive_coordinator"),
            "defensive_coordinator": parsed["raw"].get("defensive_coordinator"),
            "special_teams_coordinator": parsed["raw"].get("special_teams_coordinator"),
            "assistant_head_coach": parsed["raw"].get("assistant_head_coach"),
            "notes": "",
            "source_url": f"https://www.pro-football-reference.com/teams/{abbr}/{year}.htm",
        })

        # Long rows: one per coach per role
        for role, coaches in parsed["parsed"].items():
            if not coaches:
                continue
            for c in coaches:
                long_rows.append({
                    "season": year, "team": abbr, "team_full_name": full,
                    "role": role, "coach_name": c["name"],
                    "wins": c["wins"], "losses": c["losses"], "ties": c["ties"],
                    "games_coached": c["games"],
                    "notes": "",
                    "source_url": f"https://www.pro-football-reference.com/teams/{abbr}/{year}.htm",
                })

        if i % 25 == 0:
            rate = i / max(time.time() - t0, 1)
            eta_min = (len(tasks) - i) / max(rate, 0.001) / 60
            hc = parsed["raw"].get("head_coach", "")[:40]
            print(f"  [{i:>4}/{len(tasks)}] {abbr}/{year} {status:8s} HC={hc!r} "
                  f"ETA={eta_min:.1f}min")

    # write outputs
    wide = pd.DataFrame(wide_rows).sort_values(["season", "team"])
    wide_fp = ROOT / "coaching_history.csv"
    wide.to_csv(wide_fp, index=False)

    long_df = pd.DataFrame(long_rows).sort_values(["season", "team", "role"])
    long_fp = ROOT / "coaching_history_long.csv"
    long_df.to_csv(long_fp, index=False)

    # Quality report
    elapsed_min = (time.time() - t0) / 60
    qm = []
    qm.append("# PFR coaching history — data quality report")
    qm.append("")
    qm.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    qm.append(f"Total team-seasons: {len(tasks)} (32 franchises x 1999-2024, minus 3 pre-expansion HTX)")
    qm.append(f"Fetched fresh: {fetched_count}, cache-hits: {cache_count}, "
              f"parsed OK: {parsed_ok}, parse failed: {parse_failed}")
    qm.append(f"Wall-clock: {elapsed_min:.1f} min")
    qm.append("")
    qm.append("## Field completeness by season")
    qm.append("")
    qm.append("| season | n_teams | head_coach | OC | DC | ST | AHC |")
    qm.append("|---:|---:|---:|---:|---:|---:|---:|")
    for y in YEARS:
        wy = wide[wide["season"] == y]
        n = len(wy)
        if n == 0:
            qm.append(f"| {y} | 0 | - | - | - | - | - |")
            continue
        hc = (wy["head_coach"].notna() & (wy["head_coach"] != "")).sum()
        oc = (wy["offensive_coordinator"].notna() & (wy["offensive_coordinator"] != "")).sum()
        dc = (wy["defensive_coordinator"].notna() & (wy["defensive_coordinator"] != "")).sum()
        st = (wy["special_teams_coordinator"].notna() & (wy["special_teams_coordinator"] != "")).sum()
        ahc = (wy["assistant_head_coach"].notna() & (wy["assistant_head_coach"] != "")).sum()
        qm.append(f"| {y} | {n} | {hc}/{n} | {oc}/{n} | {dc}/{n} | {st}/{n} | {ahc}/{n} |")
    qm.append("")
    qm.append("## Failures")
    qm.append("")
    if failures:
        for abbr, year, status in failures:
            qm.append(f"- {abbr}/{year}: {status}")
    else:
        qm.append("None.")
    qm.append("")
    qm.append("## Manual-review queue (midseason coaching changes)")
    qm.append("")
    splits = wide[wide["head_coach_games_split"].notna()]
    if len(splits):
        qm.append(f"{len(splits)} team-seasons with midseason HC changes:")
        qm.append("")
        for _, r in splits.iterrows():
            qm.append(f"- {r['season']} {r['team_full_name']} ({r['team']}): "
                      f"{r['head_coach_games_split']}")
    else:
        qm.append("None detected.")
    qm.append("")

    qm_fp = ROOT / "coaching_data_quality.md"
    qm_fp.write_text("\n".join(qm), encoding="utf-8")

    print()
    print("=== SUMMARY ===")
    print(f"team-seasons attempted: {len(tasks)}")
    print(f"  fetched fresh: {fetched_count}")
    print(f"  cache hits: {cache_count}")
    print(f"  parsed ok: {parsed_ok}")
    print(f"  failures: {len(failures)}")
    print(f"wide CSV: {wide_fp} ({wide_fp.stat().st_size/1024:.1f} KB, {len(wide)} rows)")
    print(f"long CSV: {long_fp} ({long_fp.stat().st_size/1024:.1f} KB, {len(long_df)} rows)")
    print(f"quality MD: {qm_fp} ({qm_fp.stat().st_size/1024:.1f} KB)")
    print(f"wall-clock: {elapsed_min:.1f} min")


if __name__ == "__main__":
    main()
