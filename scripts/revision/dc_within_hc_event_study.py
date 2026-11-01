"""DC-within-HC-tenure + HC-within-DC-tenure event study.

Sharpens defensive-authority attribution by isolating the natural experiment:

  DC-isolated events: HC_{t-1} == HC_t  AND  DC_{t-1} != DC_t
    -> if defensive scheme shifts here, DC has authority.

  HC-isolated events: HC_{t-1} != HC_t  AND  DC_{t-1} == DC_t
    -> if defensive scheme shifts here, HC is defensive author.

  Both-change events (HC and DC both change): the original L2c/2e cell — joint authorship.
  Continuous-staff (neither changes): placebo control for L2c-style autocorrelation.

Outcomes (same 7 cells as paper §5.3):
  opp_pass_rate_residual, opp_plays_residual, opp_rz_pass_rate_residual,
  opp_epa_per_play, opp_rz_td_rate_residual, opp_rz_trips_residual,
  opp_rz_sack_rate_residual

For each isolation × cell:
  observed shift = outcome_t - outcome_{t-1}  (per team-event)
  test = |observed shift mean| vs |placebo shift mean| over B=1000 random
         continuous-staff splits.

SNR discipline:
  - permutation null B=1000
  - report n, ratio (treated mean / placebo mean), p-value
  - cells with n<10 events are reported with low-n caveat
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nfl"
OUT = DATA / "sloan_revision_outputs"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = [
    "opp_pass_rate_residual",
    "opp_plays_residual",
    "opp_rz_pass_rate_residual",
    "opp_epa_per_play",
    "opp_rz_td_rate_residual",
    "opp_rz_trips_residual",
    "opp_rz_sack_rate_residual",
]

# Team abbrev cleanup — defensive_outcome_panel uses upper-case, coach_seasons uses lower
TEAM_REMAP = {"sd": "lac", "stl": "lar", "oak": "lv"}


def build_staff_wide() -> pd.DataFrame:
    cs = pd.read_parquet(DATA / "coaching" / "coach_seasons.parquet")
    cs = cs[cs["role"].isin(["head_coach", "offensive_coordinator", "defensive_coordinator"])].copy()
    cs["team"] = cs["team"].str.lower().replace(TEAM_REMAP)
    wide = (cs.pivot_table(index=["team", "season"], columns="role",
                              values="coach_name", aggfunc="first").reset_index())
    return wide


def classify_transitions(wide: pd.DataFrame) -> pd.DataFrame:
    """For each (team, season>=min+1), classify the t-1 -> t transition."""
    wide = wide.sort_values(["team", "season"]).reset_index(drop=True)
    wide["hc_prev"] = wide.groupby("team")["head_coach"].shift(1)
    wide["oc_prev"] = wide.groupby("team")["offensive_coordinator"].shift(1)
    wide["dc_prev"] = wide.groupby("team")["defensive_coordinator"].shift(1)
    wide["season_prev"] = wide.groupby("team")["season"].shift(1)
    wide["valid"] = wide["season"] == wide["season_prev"] + 1
    wide = wide[wide["valid"]].copy()
    wide["hc_change"] = wide["head_coach"] != wide["hc_prev"]
    wide["oc_change"] = wide["offensive_coordinator"] != wide["oc_prev"]
    wide["dc_change"] = wide["defensive_coordinator"] != wide["dc_prev"]
    # Classification (defensive isolation, ignoring OC):
    conds = [
        (~wide["hc_change"] & wide["dc_change"]),   # DC-isolated
        (wide["hc_change"] & ~wide["dc_change"]),   # HC-isolated
        (wide["hc_change"] & wide["dc_change"]),    # Both
        (~wide["hc_change"] & ~wide["dc_change"]),  # Continuous
    ]
    labels = ["DC_isolated", "HC_isolated", "Both_change", "Continuous"]
    wide["transition"] = np.select(conds, labels, default="Other")
    return wide


def compute_event_shifts(transitions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """For each event, compute (outcome_t - outcome_{t-1}) per outcome cell."""
    panel = panel.copy()
    panel["team"] = panel["team"].str.lower().replace(TEAM_REMAP)
    out = transitions.copy()
    for c in OUTCOMES:
        out[f"{c}_t"] = np.nan
        out[f"{c}_tm1"] = np.nan
    for i, row in out.iterrows():
        team, t, tm1 = row["team"], row["season"], row["season_prev"]
        cur = panel[(panel["team"] == team) & (panel["season"] == t)]
        prv = panel[(panel["team"] == team) & (panel["season"] == tm1)]
        if len(cur) and len(prv):
            for c in OUTCOMES:
                out.at[i, f"{c}_t"] = float(cur[c].iloc[0])
                out.at[i, f"{c}_tm1"] = float(prv[c].iloc[0])
    for c in OUTCOMES:
        out[f"{c}_shift"] = out[f"{c}_t"] - out[f"{c}_tm1"]
    return out


def permutation_test(treated: np.ndarray, control: np.ndarray, B: int = 1000, seed: int = 42):
    """Two-sided permutation test on |mean shift| difference."""
    rng = np.random.default_rng(seed)
    treated = treated[~np.isnan(treated)]
    control = control[~np.isnan(control)]
    n_t, n_c = len(treated), len(control)
    if n_t < 3 or n_c < 3:
        return float("nan"), float("nan"), n_t, n_c
    obs_diff = abs(treated.mean()) - abs(control.mean())
    combined = np.concatenate([treated, control])
    null = np.empty(B)
    for b in range(B):
        idx = rng.permutation(len(combined))
        t_perm = combined[idx[:n_t]]
        c_perm = combined[idx[n_t:n_t + n_c]]
        null[b] = abs(t_perm.mean()) - abs(c_perm.mean())
    p_two_sided = (np.abs(null) >= abs(obs_diff)).mean()
    return obs_diff, p_two_sided, n_t, n_c


def event_test(isolation_label: str, transitions: pd.DataFrame, cell: str) -> dict:
    cell_shift = f"{cell}_shift"
    treated = transitions[transitions["transition"] == isolation_label][cell_shift].values
    cont = transitions[transitions["transition"] == "Continuous"][cell_shift].values
    treated = treated[~np.isnan(treated)]
    cont = cont[~np.isnan(cont)]
    if len(treated) < 3 or len(cont) < 3:
        return {"cell": cell, "isolation": isolation_label, "n_treated": len(treated), "n_control": len(cont),
                "treated_mean_abs": float("nan"), "control_mean_abs": float("nan"),
                "ratio_treated_to_control": float("nan"), "p_perm": float("nan")}
    obs_diff, p, n_t, n_c = permutation_test(treated, cont, B=1000, seed=42)
    t_abs = abs(treated.mean()); c_abs = abs(cont.mean())
    ratio = t_abs / c_abs if c_abs > 1e-9 else float("inf")
    return {
        "cell": cell, "isolation": isolation_label,
        "n_treated": int(n_t), "n_control": int(n_c),
        "treated_mean_abs": float(t_abs), "control_mean_abs": float(c_abs),
        "ratio_treated_to_control": float(ratio),
        "p_perm": float(p),
        "treated_mean_signed": float(treated.mean()),
        "treated_sd": float(treated.std(ddof=1)),
    }


def main():
    print("=" * 80, flush=True)
    print("Sloan Revision #2 — Defensive authority isolation event-study", flush=True)
    print("=" * 80, flush=True)

    wide = build_staff_wide()
    print(f"\nStaff wide panel: {len(wide)} (team, season) rows", flush=True)
    transitions = classify_transitions(wide)
    print(f"Valid 1-year transitions: {len(transitions)}", flush=True)

    counts = transitions["transition"].value_counts()
    print(f"\nTransition counts:", flush=True)
    for k, v in counts.items():
        print(f"  {k}: {v}", flush=True)

    panel = pd.read_parquet(DATA / "defensive_outcome_panel.parquet")
    print(f"\nDefensive outcome panel: {len(panel)} rows, seasons {panel['season'].min()}-{panel['season'].max()}", flush=True)

    events = compute_event_shifts(transitions, panel)
    events_valid = events.dropna(subset=[f"{OUTCOMES[0]}_shift"]).copy()
    print(f"\nEvents with computable shifts (in defensive panel range): {len(events_valid)}", flush=True)
    counts_valid = events_valid["transition"].value_counts()
    print(f"\nValid event counts (after defensive-panel join):", flush=True)
    for k, v in counts_valid.items():
        print(f"  {k}: {v}", flush=True)

    # Save the event panel
    events_valid.to_parquet(OUT / "dc_within_hc_events_panel.parquet")
    print(f"\nSaved event panel -> {OUT / 'dc_within_hc_events_panel.parquet'}", flush=True)

    # === Run tests ===
    results = []
    for isolation in ["DC_isolated", "HC_isolated", "Both_change"]:
        for cell in OUTCOMES:
            r = event_test(isolation, events_valid, cell)
            results.append(r)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "dc_within_hc_results.csv", index=False)
    print(f"\nSaved results -> {OUT / 'dc_within_hc_results.csv'}", flush=True)

    # === Pretty print ===
    print("\n" + "=" * 100, flush=True)
    print("RESULTS: ratio = |treated mean shift| / |continuous-staff mean shift|", flush=True)
    print("        p_perm = two-sided permutation B=1000 against continuous-staff control", flush=True)
    print("=" * 100, flush=True)

    for isolation in ["DC_isolated", "HC_isolated", "Both_change"]:
        print(f"\n--- {isolation} ---", flush=True)
        sub = results_df[results_df["isolation"] == isolation].copy()
        sub = sub.sort_values("ratio_treated_to_control", ascending=False)
        for _, row in sub.iterrows():
            verdict = " (SIG)" if row["p_perm"] < 0.05 and row["n_treated"] >= 10 else ""
            print(f"  {row['cell']:32s}  n={int(row['n_treated']):3d}  ratio={row['ratio_treated_to_control']:.2f}x  p={row['p_perm']:.3f}{verdict}", flush=True)

    # Summary headline
    print("\n" + "=" * 100, flush=True)
    print("HEADLINE — which isolation hits signal on which cells?", flush=True)
    print("=" * 100, flush=True)
    for cell in OUTCOMES:
        dc = results_df[(results_df["isolation"] == "DC_isolated") & (results_df["cell"] == cell)].iloc[0] if len(results_df[(results_df["isolation"] == "DC_isolated") & (results_df["cell"] == cell)]) else None
        hc = results_df[(results_df["isolation"] == "HC_isolated") & (results_df["cell"] == cell)].iloc[0] if len(results_df[(results_df["isolation"] == "HC_isolated") & (results_df["cell"] == cell)]) else None
        both = results_df[(results_df["isolation"] == "Both_change") & (results_df["cell"] == cell)].iloc[0] if len(results_df[(results_df["isolation"] == "Both_change") & (results_df["cell"] == cell)]) else None
        print(f"  {cell:32s}  DC-iso ratio={dc['ratio_treated_to_control']:.2f}x (p={dc['p_perm']:.3f}, n={int(dc['n_treated'])}) | "
              f"HC-iso {hc['ratio_treated_to_control']:.2f}x (p={hc['p_perm']:.3f}, n={int(hc['n_treated'])}) | "
              f"Both {both['ratio_treated_to_control']:.2f}x (p={both['p_perm']:.3f}, n={int(both['n_treated'])})", flush=True)


if __name__ == "__main__":
    main()
