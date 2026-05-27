"""Coach fingerprint clustering — Sloan paper deliverable.

Builds per-HC and per-OC schematic profile vectors pooled over tenure, clusters
each with k-means (silhouette-optimal k in [4..6]) on z-scored features, names
clusters from centroid bookends, and emits PCA scatters, radars, assignment
CSVs and a results doc.

Features (per-play means, pooled over all team-seasons of tenure):
  shotgun_rate, no_huddle_rate, deep_throw_rate, pass_rate,
  rz_shotgun_rate, rz_pass_rate

Min tenure floor: 32 team-games (~2 seasons).

Inputs:
  data/nfl/games/pbp.parquet
  data/nfl/coaching/coach_seasons.parquet

Outputs:
  figures/hc_fingerprint_pca_2d.png
  figures/hc_fingerprint_radar.png
  figures/oc_fingerprint_pca_2d.png
  figures/oc_fingerprint_radar.png
  data/nfl/hc_fingerprint_clusters.csv
  data/nfl/oc_fingerprint_clusters.csv
  docs/coach_fingerprint_clustering.md
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
NFL = ROOT / "data" / "nfl"
FIGDIR = ROOT / "figures"
DOCDIR = ROOT / "docs"
FIGDIR.mkdir(parents=True, exist_ok=True)
DOCDIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "shotgun_rate", "no_huddle_rate", "deep_throw_rate",
    "pass_rate", "rz_shotgun_rate", "rz_pass_rate",
]
PRETTY = {
    "shotgun_rate": "Shotgun",
    "no_huddle_rate": "No-Huddle",
    "deep_throw_rate": "Deep (AY>=20)",
    "pass_rate": "Pass",
    "rz_shotgun_rate": "RZ Shotgun",
    "rz_pass_rate": "RZ Pass",
}
MIN_TEAM_GAMES = 32
K_RANGE = list(range(4, 7))  # 4, 5, 6
RNG = 42

PFR_TO_NFL = {
    "rai": "LV", "rav": "BAL", "oti": "TEN", "sdg": "LAC", "htx": "HOU",
    "clt": "IND", "crd": "ARI", "ram": "LA", "buf": "BUF", "mia": "MIA",
    "nwe": "NE", "nyj": "NYJ", "cin": "CIN", "cle": "CLE", "pit": "PIT",
    "jax": "JAX", "den": "DEN", "kan": "KC", "dal": "DAL", "nyg": "NYG",
    "phi": "PHI", "was": "WAS", "chi": "CHI", "det": "DET", "gnb": "GB",
    "min": "MIN", "atl": "ATL", "car": "CAR", "nor": "NO", "tam": "TB",
    "sea": "SEA", "sfo": "SF",
}
LEGACY_REMAP = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA"}

# Famous coaches to label on PCA scatter (HCs)
HC_FAMOUS = {
    "Bill Belichick", "Sean McVay", "Kyle Shanahan", "Andy Reid",
    "Mike Tomlin", "John Harbaugh", "Pete Carroll", "Mike McCarthy",
    "Sean Payton", "Mike Vrabel", "Doug Pederson", "Matt LaFleur",
    "Zac Taylor", "Brian Daboll", "Nick Sirianni", "Dan Campbell",
    "Sean McDermott", "Bruce Arians", "Jon Gruden", "Marvin Lewis",
    "Tom Coughlin", "Jeff Fisher", "Mike Shanahan", "Bill Cowher",
    "Tony Dungy", "Mike Holmgren", "Dick Vermeil", "Brian Flores",
    "Frank Reich", "Adam Gase", "Doug Marrone", "Vic Fangio",
    "Matt Nagy", "Kliff Kingsbury", "Robert Saleh", "Arthur Smith",
    "Kevin Stefanski", "Mike Zimmer", "Mike Mularkey", "Ron Rivera",
}
# Famous OCs
OC_FAMOUS = {
    "Kyle Shanahan", "Sean McVay", "Eric Bieniemy", "Brian Daboll",
    "Bill O'Brien", "Norv Turner", "Josh McDaniels", "Greg Roman",
    "Frank Reich", "Pep Hamilton", "Mike Shula", "Kellen Moore",
    "Joe Brady", "Brian Schottenheimer", "Ben McAdoo", "Adam Gase",
    "Pat Shurmur", "Todd Haley", "Chan Gailey", "Bruce Arians",
    "Tom Clements", "Mike McCoy", "Doug Pederson", "Hue Jackson",
    "Greg Knapp", "Rob Chudzinski", "Cam Cameron", "Brian Callahan",
    "Ken Dorsey", "Matt Canada", "Brian Johnson", "Press Taylor",
    "Mike Kafka", "Luke Getsy", "Wes Phillips", "Joe Lombardi",
    "Bobby Slowik", "Mike LaFleur", "Anthony Lynn",
}


# --------------------------------------------------------------------------- #
# 1. Build per-team-season feature panel
# --------------------------------------------------------------------------- #
def build_team_season_features() -> pd.DataFrame:
    """Per (team, season) rates on the 6 features."""
    cols = ["play_id", "season", "posteam", "play_type", "shotgun",
            "no_huddle", "air_yards", "yardline_100", "qb_kneel", "qb_spike",
            "two_point_attempt"]
    pbp = pd.read_parquet(NFL / "games" / "pbp.parquet", columns=cols)
    pbp = pbp[(pbp["play_type"].isin(["pass", "run"]))
              & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)
              & (pbp["two_point_attempt"] != 1)
              & pbp["posteam"].notna()].copy()
    pbp["is_pass"] = (pbp["play_type"] == "pass").astype(int)
    pbp["is_shotgun"] = pbp["shotgun"].fillna(0).astype(int)
    pbp["is_no_huddle"] = pbp["no_huddle"].fillna(0).astype(int)
    pbp["is_deep"] = ((pbp["air_yards"].fillna(-1) >= 20)
                       & (pbp["play_type"] == "pass")).astype(int)
    pbp["is_rz"] = (pbp["yardline_100"].fillna(99) <= 20).astype(int)

    # All-down feature aggregates per team-season
    g_all = pbp.groupby(["posteam", "season"]).agg(
        n_plays=("play_id", "count"),
        pass_rate=("is_pass", "mean"),
        shotgun_rate=("is_shotgun", "mean"),
        no_huddle_rate=("is_no_huddle", "mean"),
        deep_throw_rate=("is_deep", "mean"),
    ).reset_index()

    # RZ-specific
    rz = pbp[pbp["is_rz"] == 1].copy()
    g_rz = rz.groupby(["posteam", "season"]).agg(
        n_rz_plays=("play_id", "count"),
        rz_pass_rate=("is_pass", "mean"),
        rz_shotgun_rate=("is_shotgun", "mean"),
    ).reset_index()

    out = g_all.merge(g_rz, on=["posteam", "season"], how="left")
    # Canonicalize team codes
    out["team_canon"] = out["posteam"].map(LEGACY_REMAP).fillna(out["posteam"])
    return out


# --------------------------------------------------------------------------- #
# 2. Build team-season -> (HC, OC) map with weights
# --------------------------------------------------------------------------- #
def _explode_compound_names(df: pd.DataFrame) -> pd.DataFrame:
    """Some coach_name cells contain compound entries like
    'Joe Brady , Ken Dorsey' (mid-season replacement). Split on comma into
    one row per coach, divide the assumed games equally across the splits."""
    rows = []
    for _, r in df.iterrows():
        parts = [p.strip() for p in str(r["coach_name"]).split(",") if p.strip()]
        if not parts:
            continue
        n = len(parts)
        for p in parts:
            new = r.copy()
            new["coach_name"] = p
            new["_split_n"] = n
            rows.append(new)
    return pd.DataFrame(rows)


def build_staff_map() -> pd.DataFrame:
    """For each (team, season, role) emit ALL coaches that held the role with
    their games-played share (so mid-season firings get partial credit).

    HC: uses the populated `games` field directly.
    OC: `games` is unpopulated in the data; assume 17 games per (team, season),
        split evenly across comma-separated coach entries.
    """
    cs = pd.read_parquet(NFL / "coaching" / "coach_seasons.parquet")
    cs = cs[cs["role"].isin(["head_coach", "offensive_coordinator"])].copy()
    cs["team_nfl"] = cs["team"].map(PFR_TO_NFL)
    cs = cs[cs["team_nfl"].notna()].copy()

    # Split coach_name compounds
    cs = _explode_compound_names(cs)

    # Games attribution
    season_games = {s: (17 if s >= 2021 else 16) for s in cs["season"].unique()}
    def attribute_games(row):
        if pd.notna(row.get("games", np.nan)):
            return float(row["games"]) / row["_split_n"]
        # OC: assume full season split evenly across compound entries
        return season_games[row["season"]] / row["_split_n"]
    cs["games_filled"] = cs.apply(attribute_games, axis=1)

    # Compute share within each (team_nfl, season, role)
    tot = cs.groupby(["team_nfl", "season", "role"])["games_filled"].transform("sum")
    cs["share"] = np.where(tot > 0, cs["games_filled"] / tot, 0.0)
    cs = cs[cs["share"] > 0].copy()
    return cs[["coach_name", "team_nfl", "season", "role", "games_filled", "share"]]


# --------------------------------------------------------------------------- #
# 3. Pool per-coach profile via games-weighted mean across tenure
# --------------------------------------------------------------------------- #
def build_coach_profiles(role: str, ts_feat: pd.DataFrame, staff: pd.DataFrame) -> pd.DataFrame:
    """For each coach in the given role, weight each (team, season) by their
    share*team_games and compute a games-weighted mean for each feature.

    Returns: coach_name, n_team_games, n_team_seasons + FEATURES columns.
    """
    s = staff[staff["role"] == role].copy()
    m = s.merge(ts_feat, left_on=["team_nfl", "season"],
                right_on=["team_canon", "season"], how="inner")
    if m.empty:
        return pd.DataFrame(columns=["coach_name", "n_team_games"] + FEATURES)

    # Weights: games_filled (already absolute count of games attributed to that coach)
    m["w"] = m["games_filled"]

    # For RZ features, weight should reflect rz_plays availability; we still
    # use games-weight (paper consistency) but drop rows missing rz data per feature
    rows = []
    for coach, g in m.groupby("coach_name"):
        n_team_games = float(g["w"].sum())
        n_team_seasons = int(g[["team_nfl", "season"]].drop_duplicates().shape[0])
        rec = {"coach_name": coach,
               "n_team_games": n_team_games,
               "n_team_seasons": n_team_seasons}
        for feat in FEATURES:
            sub = g[g[feat].notna()]
            if sub["w"].sum() <= 0:
                rec[feat] = np.nan
            else:
                rec[feat] = float(np.average(sub[feat], weights=sub["w"]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out


# --------------------------------------------------------------------------- #
# 4. Cluster with silhouette-optimal k
# --------------------------------------------------------------------------- #
def pick_k_and_cluster(X_std: np.ndarray, k_range=K_RANGE) -> dict:
    """Try each k in k_range, return dict with best k by silhouette plus all
    silhouette scores tried."""
    scores = {}
    fits = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RNG, n_init=20)
        labels = km.fit_predict(X_std)
        if len(set(labels)) < 2:
            scores[k] = float("nan")
            continue
        sc = silhouette_score(X_std, labels)
        scores[k] = sc
        fits[k] = (km, labels)
    best_k = max(scores, key=lambda kk: scores[kk] if not np.isnan(scores[kk]) else -1)
    km, labels = fits[best_k]
    return {"best_k": best_k, "scores": scores, "kmeans": km, "labels": labels}


# --------------------------------------------------------------------------- #
# 5. Name clusters from centroid bookends
# --------------------------------------------------------------------------- #
def label_clusters(centroids_z: np.ndarray, feat_names=FEATURES) -> dict[int, str]:
    """For each cluster (row of centroids_z), pick top-2 highest and top-1
    lowest z-score dimensions as descriptive bookends, build a label string."""
    labels = {}
    for i, row in enumerate(centroids_z):
        order = np.argsort(row)  # ascending
        lo_idx = order[0]
        hi_idx2 = order[-1]
        hi_idx1 = order[-2]

        def tag(idx, direction):
            mag = abs(row[idx])
            base = PRETTY[feat_names[idx]]
            if direction == "high":
                qual = "heavy" if mag >= 0.7 else "lean"
                return f"{base}-{qual}"
            qual = "averse" if mag >= 0.7 else "light"
            return f"{base}-{qual}"

        parts = []
        if row[hi_idx2] > 0.25:
            parts.append(tag(hi_idx2, "high"))
        if row[hi_idx1] > 0.25 and hi_idx1 != hi_idx2:
            parts.append(tag(hi_idx1, "high"))
        if row[lo_idx] < -0.25:
            parts.append(tag(lo_idx, "low"))
        if not parts:
            parts.append("League-average")
        labels[i] = " + ".join(parts)
    return labels


# --------------------------------------------------------------------------- #
# 6. Plots
# --------------------------------------------------------------------------- #
def plot_pca(profiles: pd.DataFrame, X_std: np.ndarray, labels: np.ndarray,
             cluster_names: dict[int, str], famous: set, outpath: Path, title: str):
    pca = PCA(n_components=2, random_state=RNG)
    Y = pca.fit_transform(X_std)
    fig, ax = plt.subplots(figsize=(13, 9))
    palette = sns.color_palette("Set1", n_colors=len(cluster_names))

    for cid in sorted(cluster_names):
        mask = labels == cid
        ax.scatter(Y[mask, 0], Y[mask, 1], s=70, alpha=0.7,
                   color=palette[cid], edgecolor="black", linewidth=0.4,
                   label=f"C{cid}: {cluster_names[cid]}")

    # Label famous coaches
    coaches = profiles["coach_name"].tolist()
    for j, name in enumerate(coaches):
        if name in famous:
            ax.annotate(name, (Y[j, 0], Y[j, 1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8, fontweight="bold")

    pv = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({pv[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pv[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_radar(centroids_z: np.ndarray, cluster_names: dict[int, str],
               outpath: Path, title: str):
    n_feat = len(FEATURES)
    angles = np.linspace(0, 2*np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    palette = sns.color_palette("Set1", n_colors=len(cluster_names))

    for cid in sorted(cluster_names):
        vals = centroids_z[cid].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=palette[cid], linewidth=2,
                label=f"C{cid}: {cluster_names[cid]}")
        ax.fill(angles, vals, color=palette[cid], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([PRETTY[f] for f in FEATURES], fontsize=10)
    ax.set_title(title, pad=22)
    # Symmetric radial range
    rmax = max(2.0, float(np.abs(centroids_z).max()) + 0.3)
    ax.set_ylim(-rmax, rmax)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["-1σ", "0", "+1σ"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. Orchestrator per role
# --------------------------------------------------------------------------- #
def run_for_role(role: str, ts_feat: pd.DataFrame, staff: pd.DataFrame,
                 famous: set, role_label: str) -> dict:
    profiles = build_coach_profiles(role, ts_feat, staff)
    profiles_full = profiles.copy()
    # Tenure floor
    profiles = profiles[profiles["n_team_games"] >= MIN_TEAM_GAMES].copy()
    # Drop rows with any feature NaN (rare, RZ data should be there)
    profiles = profiles.dropna(subset=FEATURES).reset_index(drop=True)

    n_before = len(profiles_full)
    n_after = len(profiles)
    print(f"[{role_label}] {n_before} candidates, {n_after} pass {MIN_TEAM_GAMES}-team-game floor")

    X = profiles[FEATURES].to_numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    cl = pick_k_and_cluster(X_std)
    k = cl["best_k"]
    labels = cl["labels"]
    centroids_z = cl["kmeans"].cluster_centers_  # in z-space

    cluster_names = label_clusters(centroids_z)
    profiles["cluster_id"] = labels
    profiles["cluster_label"] = profiles["cluster_id"].map(cluster_names)

    return {
        "profiles": profiles,
        "X_std": X_std,
        "labels": labels,
        "scaler": scaler,
        "centroids_z": centroids_z,
        "cluster_names": cluster_names,
        "best_k": k,
        "scores": cl["scores"],
        "n_before_floor": n_before,
    }


# --------------------------------------------------------------------------- #
# 8. Docs
# --------------------------------------------------------------------------- #
def write_docs(hc_res: dict, oc_res: dict, outpath: Path):
    league_means = {}  # filled below
    lines = []
    lines.append("# Coach fingerprint clustering — Sloan paper deliverable\n")
    lines.append("## Method\n")
    lines.append(f"- Features (per-play rates pooled over tenure, games-weighted): "
                 f"{', '.join(FEATURES)}")
    lines.append(f"- Tenure floor: ≥ {MIN_TEAM_GAMES} team-games (~ 2 seasons)")
    lines.append("- k-means on z-scored features; k chosen by silhouette over "
                 f"{K_RANGE}")
    lines.append("- Cluster labels derived from centroid bookends in z-space "
                 "(top-2 high + top-1 low, magnitude-keyed)")
    lines.append(f"- Data: pbp.parquet seasons 1999-2025; coach_seasons.parquet 1999-2024")
    lines.append("")

    for tag, res in [("Head Coaches", hc_res), ("Offensive Coordinators", oc_res)]:
        lines.append(f"## {tag}\n")
        scores = res["scores"]
        lines.append(f"- N coaches passing tenure floor: **{len(res['profiles'])}** "
                     f"(of {res['n_before_floor']} candidates)")
        lines.append(f"- Silhouette scores by k:")
        for k in sorted(scores):
            mark = " ← chosen" if k == res["best_k"] else ""
            sv = scores[k]
            lines.append(f"  - k={k}: {sv:.4f}{mark}")
        lines.append(f"- Chosen k: **{res['best_k']}**")
        lines.append("")

        # Cluster summaries
        prof = res["profiles"]
        lines.append("### Clusters (z-score centroids on each feature)\n")
        lines.append("| Cluster | Label | n | " +
                     " | ".join(PRETTY[f] for f in FEATURES) + " |")
        lines.append("|" + "---|"*(len(FEATURES) + 3))
        for cid in sorted(res["cluster_names"]):
            row = res["centroids_z"][cid]
            n_in = int((res["labels"] == cid).sum())
            cells = " | ".join(f"{v:+.2f}" for v in row)
            lines.append(f"| C{cid} | {res['cluster_names'][cid]} | {n_in} | {cells} |")
        lines.append("")

        # Cluster raw-rate centroids
        lines.append("### Cluster mean raw rates\n")
        lines.append("| Cluster | Label | " +
                     " | ".join(PRETTY[f] for f in FEATURES) + " |")
        lines.append("|" + "---|"*(len(FEATURES) + 2))
        for cid in sorted(res["cluster_names"]):
            mask = res["labels"] == cid
            row = prof.loc[mask, FEATURES].mean()
            cells = " | ".join(f"{row[f]*100:.1f}%" for f in FEATURES)
            lines.append(f"| C{cid} | {res['cluster_names'][cid]} | {cells} |")
        lines.append("")

        # Representative coaches (most team-games per cluster, up to 10)
        lines.append("### Representative coaches per cluster (up to 10, by tenure)\n")
        for cid in sorted(res["cluster_names"]):
            mask = prof["cluster_id"] == cid
            reps = (prof.loc[mask].sort_values("n_team_games", ascending=False)
                    .head(10)["coach_name"].tolist())
            lines.append(f"- **C{cid} ({res['cluster_names'][cid]})** "
                         f"(n={int(mask.sum())}): {', '.join(reps)}")
        lines.append("")

    lines.append("## Reproduce\n")
    lines.append("```bash")
    lines.append("python \"scripts/coach_fingerprint_clustering.py\"")
    lines.append("```")
    lines.append("")
    lines.append("Outputs:")
    lines.append("- `figures/hc_fingerprint_pca_2d.png`")
    lines.append("- `figures/hc_fingerprint_radar.png`")
    lines.append("- `figures/oc_fingerprint_pca_2d.png`")
    lines.append("- `figures/oc_fingerprint_radar.png`")
    lines.append("- `data/nfl/hc_fingerprint_clusters.csv`")
    lines.append("- `data/nfl/oc_fingerprint_clusters.csv`")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print("=== building team-season feature panel from pbp ===")
    ts_feat = build_team_season_features()
    print(f"  {len(ts_feat):,} team-season rows")
    print("=== building staff map (HC + OC) ===")
    staff = build_staff_map()
    print(f"  {len(staff):,} (coach, team, season, role) rows")

    print("=== HC ===")
    hc_res = run_for_role("head_coach", ts_feat, staff, HC_FAMOUS, "HC")
    print(f"  best k = {hc_res['best_k']}  scores = "
          + ", ".join(f"k={k}:{v:.3f}" for k, v in hc_res["scores"].items()))
    print("=== OC ===")
    oc_res = run_for_role("offensive_coordinator", ts_feat, staff, OC_FAMOUS, "OC")
    print(f"  best k = {oc_res['best_k']}  scores = "
          + ", ".join(f"k={k}:{v:.3f}" for k, v in oc_res["scores"].items()))

    # ---- plots ----
    print("=== plotting HC ===")
    plot_pca(hc_res["profiles"], hc_res["X_std"], hc_res["labels"],
             hc_res["cluster_names"], HC_FAMOUS,
             FIGDIR / "hc_fingerprint_pca_2d.png",
             title=f"HC schematic fingerprint — PCA 2D (k={hc_res['best_k']})")
    plot_radar(hc_res["centroids_z"], hc_res["cluster_names"],
               FIGDIR / "hc_fingerprint_radar.png",
               title=f"HC cluster centroids — radar (k={hc_res['best_k']}, z-score)")

    print("=== plotting OC ===")
    plot_pca(oc_res["profiles"], oc_res["X_std"], oc_res["labels"],
             oc_res["cluster_names"], OC_FAMOUS,
             FIGDIR / "oc_fingerprint_pca_2d.png",
             title=f"OC schematic fingerprint — PCA 2D (k={oc_res['best_k']})")
    plot_radar(oc_res["centroids_z"], oc_res["cluster_names"],
               FIGDIR / "oc_fingerprint_radar.png",
               title=f"OC cluster centroids — radar (k={oc_res['best_k']}, z-score)")

    # ---- csvs ----
    print("=== writing CSVs ===")
    out_cols = ["coach_name", "n_team_games", "cluster_id", "cluster_label"] + FEATURES
    hc_csv = NFL / "hc_fingerprint_clusters.csv"
    oc_csv = NFL / "oc_fingerprint_clusters.csv"
    (hc_res["profiles"][out_cols]
        .sort_values(["cluster_id", "n_team_games"], ascending=[True, False])
        .to_csv(hc_csv, index=False))
    (oc_res["profiles"][out_cols]
        .sort_values(["cluster_id", "n_team_games"], ascending=[True, False])
        .to_csv(oc_csv, index=False))
    print(f"  {hc_csv}")
    print(f"  {oc_csv}")

    # ---- doc ----
    print("=== writing docs ===")
    write_docs(hc_res, oc_res, DOCDIR / "coach_fingerprint_clustering.md")
    print(f"  {DOCDIR / 'coach_fingerprint_clustering.md'}")

    # ---- summary print ----
    print("\n=== summary ===")
    print(f"HC: n={len(hc_res['profiles'])}  k={hc_res['best_k']}")
    for cid, name in hc_res["cluster_names"].items():
        n_in = int((hc_res["labels"] == cid).sum())
        print(f"  C{cid} ({n_in}): {name}")
    print(f"OC: n={len(oc_res['profiles'])}  k={oc_res['best_k']}")
    for cid, name in oc_res["cluster_names"].items():
        n_in = int((oc_res["labels"] == cid).sum())
        print(f"  C{cid} ({n_in}): {name}")


if __name__ == "__main__":
    main()
