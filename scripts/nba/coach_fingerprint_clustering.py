"""NBA HC fingerprint clustering — k-means on schematic profile vectors.

Mirrors NFL coach_fingerprint_clustering. Builds per-HC profile of average
schematic-residual tendencies across all their seasons in the window, then
k-means with silhouette scoring to find natural archetypes.

Features per HC (each is a z-scored mean of season-aggregated residual):
  pace_resid                  - tempo identity
  three_pt_rate_resid         - shot selection (own)
  ast_rate_resid              - ball-movement / offensive style
  opp_three_pt_rate_resid     - perimeter defensive scheme (REAL coach effect per propensity)
  opp_ast_rate_resid          - ball-movement allowed (defensive)

Output:
  data/results/coach_fingerprints.csv (per-HC profile + cluster assignment)
  figures/hc_pca_2d.png (PCA 2D scatter colored by cluster)
  figures/hc_cluster_radars.png (per-cluster radar of feature means)
  docs/coach_fingerprint_clustering.md (summary writeup)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2] / "data" / "nba"
NBA_PROJ = Path(__file__).resolve().parents[2] / "data" / "nba" / "inputs"   # staged per mode: scripts/build/stage_nba.py
RESULTS = ROOT / "data" / "results"
FIG = ROOT / "figures"
DOCS = ROOT / "docs"
RESULTS.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MIN_SEASONS = 2  # require HC with at least 2 seasons in window
SEED = 42


def build_team_season_residuals():
    """Season-aggregate residuals (mean game-level residual per team-season)."""
    bs = pd.read_parquet(NBA_PROJ / "historical_box_scores.parquet")
    bs = bs[bs["season_type"] == "Regular Season"].copy()
    tg = (bs.groupby(["game_id", "season", "team_abbr", "game_date"], as_index=False)
            .agg(FGM=("FGM", "sum"), FGA=("FGA", "sum"),
                 FG3A=("FG3A", "sum"), FTA=("FTA", "sum"),
                 OREB=("OREB", "sum"), AST=("AST", "sum"),
                 TOV=("TOV", "sum")))
    tg["possessions"] = tg["FGA"] - tg["OREB"] + tg["TOV"] + 0.44 * tg["FTA"]
    tg["three_pt_rate"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    tg["ast_rate"] = tg["AST"] / tg["FGM"].clip(lower=1)
    opp = tg.rename(columns={"team_abbr": "opp_team",
                              "three_pt_rate": "opp_three_pt_rate",
                              "ast_rate": "opp_ast_rate"})[
        ["game_id", "opp_team", "opp_three_pt_rate", "opp_ast_rate"]]
    merged = tg.merge(opp, on="game_id", how="left")
    merged = merged[merged["opp_team"] != merged["team_abbr"]].copy()
    for met in ["possessions", "three_pt_rate", "ast_rate",
                 "opp_three_pt_rate", "opp_ast_rate"]:
        smean = merged.groupby("season")[met].transform("mean")
        merged[f"{met}_resid"] = merged[met] - smean
    season_agg = (merged.groupby(["team_abbr", "season"], as_index=False)
                          .agg(pace_resid=("possessions_resid", "mean"),
                                three_pt_rate_resid=("three_pt_rate_resid", "mean"),
                                ast_rate_resid=("ast_rate_resid", "mean"),
                                opp_three_pt_rate_resid=("opp_three_pt_rate_resid", "mean"),
                                opp_ast_rate_resid=("opp_ast_rate_resid", "mean")))
    return season_agg


def attach_hc(season_agg):
    cts = pd.read_parquet(NBA_PROJ / "coach_team_season.parquet")
    cts = cts[["team_abbr", "season", "head_coach_name", "mid_season_change"]].rename(
        columns={"head_coach_name": "HC"})
    # Drop mid-season change team-seasons (ambiguous HC attribution at season level)
    cts = cts[cts["mid_season_change"] == False]
    return season_agg.merge(cts, on=["team_abbr", "season"], how="inner").dropna(subset=["HC"])


def aggregate_per_hc(season_hc):
    feats = ["pace_resid", "three_pt_rate_resid", "ast_rate_resid",
              "opp_three_pt_rate_resid", "opp_ast_rate_resid"]
    per_hc = (season_hc.groupby("HC", as_index=False)
                        .agg({f: "mean" for f in feats}))
    per_hc["n_seasons"] = season_hc.groupby("HC").size().values
    per_hc = per_hc[per_hc["n_seasons"] >= MIN_SEASONS].reset_index(drop=True)
    return per_hc, feats


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def main():
    print("=== NBA HC FINGERPRINT CLUSTERING ===\n")
    season_agg = build_team_season_residuals()
    print(f"team-season residual rows: {len(season_agg):,}")
    season_hc = attach_hc(season_agg)
    print(f"  with HC attached (excluding mid-season-change seasons): {len(season_hc):,}")
    per_hc, feats = aggregate_per_hc(season_hc)
    print(f"  HCs with >= {MIN_SEASONS} seasons in window: {len(per_hc)}\n")

    X = standardize(per_hc[feats].values)

    # Silhouette across k
    print(f"  {'k':>3} {'silhouette':>12} {'inertia':>10}")
    sil_results = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_)
        sil_results[k] = (sil, km)
        print(f"  {k:>3} {sil:>12.4f} {km.inertia_:>10.2f}")
    best_k = max(sil_results.items(), key=lambda x: x[1][0])[0]
    best_sil, km = sil_results[best_k]
    per_hc["cluster"] = km.labels_
    print(f"\n  -> BEST k={best_k} (silhouette={best_sil:.4f})")

    # NFL ref
    print(f"\n  NFL reference: HC k=4 (silhouettes 0.31-0.34), OC k=5 (0.28-0.29)")

    # Cluster profiles
    cluster_means = per_hc.groupby("cluster", as_index=False).agg(
        n=("HC", "count"), **{f: (f, "mean") for f in feats})
    print(f"\n  Cluster centers (mean residual per feature):")
    print(cluster_means.to_string(index=False))

    # PCA 2D
    pca = PCA(n_components=2)
    XY = pca.fit_transform(X)
    per_hc["pc1"] = XY[:, 0]; per_hc["pc2"] = XY[:, 1]
    print(f"\n  PCA explained variance: pc1={pca.explained_variance_ratio_[0]:.3f}, "
          f"pc2={pca.explained_variance_ratio_[1]:.3f}")

    # Output CSV
    per_hc.to_csv(RESULTS / "coach_fingerprints.csv", index=False)
    print(f"\nwrote: {RESULTS / 'coach_fingerprints.csv'}")

    # PCA scatter figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab10.colors
    for c in sorted(per_hc["cluster"].unique()):
        sub = per_hc[per_hc["cluster"] == c]
        ax.scatter(sub["pc1"], sub["pc2"], c=[colors[c]], label=f"Cluster {c} (n={len(sub)})",
                    alpha=0.7, s=80, edgecolors="black", linewidth=0.5)
        # Annotate notable coaches (more than 5 seasons, or extreme positions)
        for _, r in sub.iterrows():
            if r["n_seasons"] >= 5 or abs(r["pc1"]) > 1.5 or abs(r["pc2"]) > 1.5:
                ax.annotate(r["HC"].split()[-1], (r["pc1"], r["pc2"]),
                            fontsize=7, alpha=0.75)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"NBA HC Schematic Fingerprints (k={best_k}, sil={best_sil:.3f})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "hc_pca_2d.png", dpi=130)
    plt.close()
    print(f"wrote: {FIG / 'hc_pca_2d.png'}")

    # Radar per cluster
    feat_labels = ["pace", "3PT rate", "AST rate", "opp 3PT", "opp AST"]
    angles = np.linspace(0, 2 * np.pi, len(feat_labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, best_k, figsize=(4 * best_k, 4),
                              subplot_kw=dict(polar=True))
    if best_k == 1:
        axes = [axes]
    for c in range(best_k):
        ax = axes[c]
        sub = per_hc[per_hc["cluster"] == c]
        vals = sub[feats].mean().values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, c=colors[c])
        ax.fill(angles, vals, alpha=0.25, color=colors[c])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feat_labels, fontsize=8)
        ax.set_title(f"Cluster {c} (n={len(sub)})", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "hc_cluster_radars.png", dpi=130)
    plt.close()
    print(f"wrote: {FIG / 'hc_cluster_radars.png'}")

    # Print top members per cluster
    print(f"\n  Members per cluster:")
    for c in sorted(per_hc["cluster"].unique()):
        sub = per_hc[per_hc["cluster"] == c].sort_values("n_seasons", ascending=False)
        names = ", ".join(sub.head(8)["HC"].tolist())
        print(f"    Cluster {c} (n={len(sub)}): {names}{'...' if len(sub) > 8 else ''}")


if __name__ == "__main__":
    main()
