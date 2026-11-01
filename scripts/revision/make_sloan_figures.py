"""Generate two leverage figures for the Sloan paper:

Figure 1 — Concept-family portability forest plot (§6.2 falsification visualized)
  Shows all 6 OC concepts landing in the 0.41-0.50 slope band regardless of bucket.
  Turns "one trick (RZ shotgun)" into "broad schematic identity."

Figure 2 — Cross-league side-by-side rigor stack (§6.5 methodological differentiator)
  Shows NFL OC RZ-shotgun + NFL DC isolated cells alongside NBA HC opp_3PT_rate
  under the same 3-layer rigor stack (naive ratio / propensity matched / predictive slope).

Drops both PNGs into C:/Users/Nate/OneDrive/Documents/Sloan/NFL/
"""
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "data" / "nfl" / "sloan_revision_outputs"
TARGET = Path(__file__).resolve().parents[2] / "figures"
TARGET.mkdir(parents=True, exist_ok=True)


def ols_slope_with_se(x, y):
    """Returns (slope, se_slope, r) for y ~ a + b*x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float((resid ** 2).sum()) / (n - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    var_beta = sigma2 * XtX_inv
    se_slope = float(np.sqrt(var_beta[1, 1]))
    r = float(np.corrcoef(x, y)[0, 1])
    return float(beta[1]), se_slope, r, n


# === Concept registry mirrors oc_concept_family_portability.py ===
CONCEPTS = [
    ("rz_shotgun_rate",         "Roster-INDEP", "RZ shotgun rate"),
    ("goalline_shotgun_rate",   "Roster-INDEP", "Goal-line shotgun rate (≤5 yd)"),
    ("late2_no_huddle_rate",    "Roster-INDEP", "Late-2-min no-huddle rate"),
    ("non_rz_shotgun_rate",     "Mixed",        "Non-RZ shotgun rate"),
    ("deep_shot_rate",          "Roster-COND",  "Deep-shot rate (air_yds ≥ 20)"),
    ("early_neutral_pass_rate", "Roster-COND",  "Early-down neutral pass rate"),
]

BUCKET_COLOR = {
    "Roster-INDEP": "#1f77b4",   # blue
    "Mixed":        "#7f7f7f",   # grey
    "Roster-COND":  "#d62728",   # red
}


def make_concept_forest():
    rows = []
    for concept_col, bucket, label in CONCEPTS:
        paired = pd.read_csv(OUT / f"oc_paired_{concept_col}.csv")
        slope, se, r, n = ols_slope_with_se(paired["shift_pred"], paired["shift_obs"])
        ci_lo = slope - 1.96 * se
        ci_hi = slope + 1.96 * se
        rows.append({
            "concept": concept_col, "bucket": bucket, "label": label,
            "slope": slope, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "r": r, "n": n,
        })
    df = pd.DataFrame(rows)
    # Order: Roster-INDEP (top), Mixed, Roster-COND (bottom)
    bucket_order = {"Roster-INDEP": 0, "Mixed": 1, "Roster-COND": 2}
    df["_order"] = df["bucket"].map(bucket_order)
    df = df.sort_values(["_order", "slope"], ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    y_positions = np.arange(len(df))

    # Highlighted slope band [0.41, 0.50]
    ax.add_patch(Rectangle((0.41, -0.5), 0.09, len(df), color="#fff2b0", alpha=0.6,
                              zorder=0, label="Observed band 0.41-0.50"))

    # Reference lines
    ax.axvline(0.0, color="#888", linestyle="-", linewidth=0.7, alpha=0.6, zorder=1)
    ax.axvline(1.0, color="#888", linestyle="--", linewidth=0.7, alpha=0.6, zorder=1,
                label="Perfect transfer (slope=1)")

    # Dots + CI
    for i, row in df.iterrows():
        color = BUCKET_COLOR[row["bucket"]]
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=color, linewidth=2.5,
                alpha=0.85, zorder=3)
        ax.plot(row["slope"], i, "o", color=color, markersize=11, markeredgecolor="black",
                markeredgewidth=0.7, zorder=4)
        ax.text(row["ci_hi"] + 0.03, i,
                f"n={row['n']}, r={row['r']:+.2f}",
                va="center", fontsize=8.5, color="#444")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for _, row in df.iterrows()], fontsize=10)
    ax.set_xlabel("OLS portability slope (observed shift on predicted shift)\n95% CI bars", fontsize=10.5)
    ax.set_xlim(-0.05, 1.2)
    ax.set_ylim(-0.5, len(df) - 0.5)

    # Legend — bucket swatches
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUCKET_COLOR["Roster-INDEP"],
                markersize=10, markeredgecolor="black", markeredgewidth=0.6, label="Roster-INDEP (field-position conditioned)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUCKET_COLOR["Mixed"],
                markersize=10, markeredgecolor="black", markeredgewidth=0.6, label="Mixed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUCKET_COLOR["Roster-COND"],
                markersize=10, markeredgecolor="black", markeredgewidth=0.6, label="Roster-COND (paper §6.2 named)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5, frameon=False,
              title="Concept family", title_fontsize=9)

    ax.set_title("OC schematic portability — six concepts span both paper §6.2 buckets\n"
                  "All slopes land in the 0.41-0.50 band. The strict roster-INDEP > roster-COND\n"
                  "prediction is not supported by the available data.",
                  fontsize=10.5, loc="left")

    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.invert_yaxis()

    fig.savefig(TARGET / "fig_concept_family_portability_forest.png", dpi=200, bbox_inches="tight")
    fig.savefig(TARGET / "fig_concept_family_portability_forest.pdf", bbox_inches="tight")
    print(f"Saved -> {TARGET / 'fig_concept_family_portability_forest.png'}")
    print(f"Saved -> {TARGET / 'fig_concept_family_portability_forest.pdf'}")
    plt.close(fig)
    return df


def make_cross_league_panel():
    """Cross-league side-by-side rigor stack."""
    # NFL OC RZ-shotgun (paper headline, Move #1 strengthened)
    # NFL DC opp_plays (Move #2 — DC-isolated)
    # NBA HC opp_3PT_rate (Move #4)
    #
    # Three rigor layers shown:
    #   (a) Naive event-study (ratio vs control; null=1)
    #   (b) Propensity-matched (ratio vs control; null=1)
    #   (c) Retread predictive validation (OLS/Bayes slope; null=0, perfect=1)

    rows = [
        # NFL — OC RZ-shotgun (paper headline, strengthened by Move #1)
        dict(sport="NFL", role="OC",  cell="RZ shotgun rate",
             naive_ratio=1.34, naive_p=0.0002, naive_n=175,
             matched_ratio=None,  # paper says "unaffected by matching" on slope
             matched_p=None, matched_n=None,
             # Move #1 hierarchical Bayes slope:
             slope=0.469, slope_lo=0.320, slope_hi=0.621,
             r=0.483, slope_n=151,
             matched_slope=0.439, matched_slope_lo=0.266, matched_slope_hi=0.613,
             matched_slope_n=109),
        # NFL — DC opp_plays (Move #2 — DC-isolated within HC tenure, 2017-2024 cohort)
        dict(sport="NFL", role="DC",  cell="opp_plays (DC-isolated)",
             naive_ratio=6.37, naive_p=0.007, naive_n=31,
             matched_ratio=None,
             matched_p=None, matched_n=None,
             slope=None, slope_lo=None, slope_hi=None, r=None, slope_n=None,
             matched_slope=None, matched_slope_lo=None, matched_slope_hi=None,
             matched_slope_n=None),
        # NBA — HC opp_3PT_rate (Move #4 — perimeter defensive scheme)
        dict(sport="NBA", role="HC",  cell="opp_3PT_rate",
             naive_ratio=1.21, naive_p=0.040, naive_n=61,
             matched_ratio=1.28, matched_p=0.048, matched_n=52,
             slope=0.560, slope_lo=None, slope_hi=None,
             r=0.550, slope_n=24,
             matched_slope=None, matched_slope_lo=None, matched_slope_hi=None,
             matched_slope_n=None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1, 1, 1]})

    labels = [f"{r['sport']} — {r['role']}: {r['cell']}" for r in rows]
    y_pos = np.arange(len(rows))
    colors = ["#1f77b4" if r["sport"] == "NFL" else "#d62728" for r in rows]

    # === Panel A: naive ratio ===
    ax = axes[0]
    ax.axvline(1.0, color="#888", linestyle="-", linewidth=0.7, alpha=0.6, zorder=1)
    for i, r in enumerate(rows):
        if r["naive_ratio"] is None:
            continue
        ax.plot(r["naive_ratio"], i, "o", color=colors[i], markersize=14,
                markeredgecolor="black", markeredgewidth=0.7, zorder=3)
        sig = " ★" if r["naive_p"] < 0.05 else ""
        ax.text(r["naive_ratio"], i + 0.18, f"  {r['naive_ratio']:.2f}× (p={r['naive_p']:.3f}){sig}\n  n={r['naive_n']}",
                va="bottom", ha="center", fontsize=9, color="#222")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Naive ratio vs continuous-staff control\n(null = 1.0)", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 12.0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("(A) Naive event-study", fontsize=11, loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.invert_yaxis()

    # === Panel B: propensity-matched ratio (or matched-slope where applicable) ===
    ax = axes[1]
    ax.axvline(1.0, color="#888", linestyle="-", linewidth=0.7, alpha=0.6, zorder=1)
    for i, r in enumerate(rows):
        if r["matched_ratio"] is not None:
            ax.plot(r["matched_ratio"], i, "s", color=colors[i], markersize=14,
                    markeredgecolor="black", markeredgewidth=0.7, zorder=3)
            sig = " ★" if r["matched_p"] < 0.05 else ""
            ax.text(r["matched_ratio"], i + 0.18, f"  {r['matched_ratio']:.2f}× (p={r['matched_p']:.3f}){sig}\n  n={r['matched_n']}",
                    va="bottom", ha="center", fontsize=9, color="#222")
        elif r["matched_slope"] is not None:
            # Plot slope as alternative matched estimate (for NFL OC)
            ax.plot(r["matched_slope"], i, "s", color=colors[i], markersize=14,
                    markeredgecolor="black", markeredgewidth=0.7, zorder=3)
            ax.plot([r["matched_slope_lo"], r["matched_slope_hi"]], [i, i],
                    color=colors[i], linewidth=2.0, alpha=0.7, zorder=2)
            ax.text(r["matched_slope"], i + 0.22,
                    f"  Bayes slope {r['matched_slope']:.3f}\n  [{r['matched_slope_lo']:.3f}, {r['matched_slope_hi']:.3f}]  n={r['matched_slope_n']}",
                    va="bottom", ha="center", fontsize=8.5, color="#222")
        else:
            ax.text(0.5, i, "(not tested)", va="center", ha="center", fontsize=9, color="#888", style="italic")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(rows))
    ax.set_xlabel("Propensity / covariate-adjusted estimate\n(NBA: matched ratio; NFL OC: hierarchical Bayes slope w/ covariates)", fontsize=9.5)
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("(B) Propensity-matched / adjusted", fontsize=11, loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    # === Panel C: retread predictive validation slope ===
    ax = axes[2]
    ax.axvline(0.0, color="#888", linestyle="-", linewidth=0.7, alpha=0.6, zorder=1)
    ax.axvline(1.0, color="#888", linestyle="--", linewidth=0.7, alpha=0.6, zorder=1)
    for i, r in enumerate(rows):
        if r["slope"] is None:
            ax.text(0.5, i, "(not tested)", va="center", ha="center", fontsize=9, color="#888", style="italic")
            continue
        ax.plot(r["slope"], i, "D", color=colors[i], markersize=14,
                markeredgecolor="black", markeredgewidth=0.7, zorder=3)
        if r["slope_lo"] is not None and r["slope_hi"] is not None:
            ax.plot([r["slope_lo"], r["slope_hi"]], [i, i], color=colors[i], linewidth=2.0,
                    alpha=0.7, zorder=2)
            ax.text(r["slope"], i + 0.22,
                    f"  slope {r['slope']:.3f} [95% {r['slope_lo']:.3f}, {r['slope_hi']:.3f}]\n  r={r['r']:.3f}, n={r['slope_n']}",
                    va="bottom", ha="center", fontsize=8.5, color="#222")
        else:
            ax.text(r["slope"], i + 0.18,
                    f"  slope {r['slope']:.3f}\n  r={r['r']:.3f}, n={r['slope_n']}",
                    va="bottom", ha="center", fontsize=9, color="#222")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(rows))
    ax.set_xlabel("Retread predictive-validation slope\n(null=0, perfect transfer=1.0)", fontsize=10)
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("(C) Retread predictive validation", fontsize=11, loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    fig.suptitle("Cross-league mirror — same four-layer SNR-disciplined framework, two sports, sport-specific surviving cells",
                    fontsize=12, weight="bold", y=1.03)

    fig.savefig(TARGET / "fig_cross_league_rigor_stack.png", dpi=200, bbox_inches="tight")
    fig.savefig(TARGET / "fig_cross_league_rigor_stack.pdf", bbox_inches="tight")
    print(f"Saved -> {TARGET / 'fig_cross_league_rigor_stack.png'}")
    print(f"Saved -> {TARGET / 'fig_cross_league_rigor_stack.pdf'}")
    plt.close(fig)


def main():
    print("=" * 80)
    print("Generating two leverage figures for Sloan paper revision")
    print("=" * 80)
    print("\n[Figure 1] Concept-family portability forest plot...")
    df = make_concept_forest()
    # Console print summary (ascii-safe)
    for _, row in df.iterrows():
        print(f"  {row['bucket']:12s}  slope={row['slope']:+.3f}  CI=[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]  r={row['r']:+.3f}  n={int(row['n'])}")
    print("\n[Figure 2] Cross-league rigor-stack side-by-side...")
    make_cross_league_panel()
    print("\nDone.")


if __name__ == "__main__":
    main()
