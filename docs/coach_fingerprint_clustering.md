# Coach fingerprint clustering — Sloan paper deliverable

## Method

- Features (per-play rates pooled over tenure, games-weighted): shotgun_rate, no_huddle_rate, deep_throw_rate, pass_rate, rz_shotgun_rate, rz_pass_rate
- Tenure floor: ≥ 32 team-games (~ 2 seasons)
- k-means on z-scored features; k chosen by silhouette over [4, 5, 6]
- Cluster labels derived from centroid bookends in z-space (top-2 high + top-1 low, magnitude-keyed)
- Data: pbp.parquet seasons 1999-2025; coach_seasons.parquet 1999-2024

## Head Coaches

- N coaches passing tenure floor: **129** (of 172 candidates)
- Silhouette scores by k:
  - k=4: 0.3384 ← chosen
  - k=5: 0.3141
  - k=6: 0.3293
- Chosen k: **4**

### Clusters (z-score centroids on each feature)

| Cluster | Label | n | Shotgun | No-Huddle | Deep (AY>=20) | Pass | RZ Shotgun | RZ Pass |
|---|---|---|---|---|---|---|---|---|
| C0 | Shotgun-lean + RZ Shotgun-lean + Pass-averse | 32 | +0.47 | +0.32 | +0.44 | -0.76 | +0.45 | -0.62 |
| C1 | Deep (AY>=20)-averse | 45 | -1.13 | -0.70 | -1.17 | -0.35 | -1.15 | -0.52 |
| C2 | No-Huddle-heavy + Shotgun-heavy | 3 | +1.76 | +4.71 | +0.57 | +0.26 | +1.69 | +0.13 |
| C3 | RZ Pass-heavy + Pass-heavy | 49 | +0.62 | +0.15 | +0.75 | +0.80 | +0.66 | +0.87 |

### Cluster mean raw rates

| Cluster | Label | Shotgun | No-Huddle | Deep (AY>=20) | Pass | RZ Shotgun | RZ Pass |
|---|---|---|---|---|---|---|---|
| C0 | Shotgun-lean + RZ Shotgun-lean + Pass-averse | 54.3% | 8.9% | 5.8% | 55.3% | 51.4% | 48.3% |
| C1 | Deep (AY>=20)-averse | 18.0% | 1.1% | 1.6% | 56.7% | 12.8% | 48.9% |
| C2 | No-Huddle-heavy + Shotgun-heavy | 83.5% | 42.4% | 6.1% | 58.7% | 81.5% | 52.2% |
| C3 | RZ Pass-heavy + Pass-heavy | 57.7% | 7.6% | 6.6% | 60.5% | 56.4% | 56.0% |

### Representative coaches per cluster (up to 10, by tenure)

- **C0 (Shotgun-lean + RZ Shotgun-lean + Pass-averse)** (n=32): Bill Belichick, John Harbaugh, John Fox, Marvin Lewis, Tom Coughlin, Pete Carroll, Ron Rivera, Jack Del Rio, Sean McVay, Kyle Shanahan
- **C1 (Deep (AY>=20)-averse)** (n=45): Jeff Fisher, Mike Shanahan, Jon Gruden, Lovie Smith, Tony Dungy, Mike Holmgren, Gary Kubiak, Norv Turner, Brian Billick, Dick Jauron
- **C2 (No-Huddle-heavy + Shotgun-heavy)** (n=3): Kliff Kingsbury, Chip Kelly, Joe Judge
- **C3 (RZ Pass-heavy + Pass-heavy)** (n=49): Andy Reid, Mike Tomlin, Mike McCarthy, Sean Payton, Jason Garrett, Doug Pederson, Bruce Arians, Ken Whisenhunt, Todd Bowles, Jim Caldwell

## Offensive Coordinators

- N coaches passing tenure floor: **141** (of 197 candidates)
- Silhouette scores by k:
  - k=4: 0.2852
  - k=5: 0.2933 ← chosen
  - k=6: 0.2891
- Chosen k: **5**

### Clusters (z-score centroids on each feature)

| Cluster | Label | n | Shotgun | No-Huddle | Deep (AY>=20) | Pass | RZ Shotgun | RZ Pass |
|---|---|---|---|---|---|---|---|---|
| C0 | RZ Pass-heavy + Pass-heavy | 43 | +0.75 | +0.12 | +0.76 | +0.86 | +0.78 | +0.89 |
| C1 | Pass-lean + RZ Shotgun-averse | 29 | -1.19 | -0.79 | -1.03 | +0.31 | -1.19 | +0.00 |
| C2 | Shotgun-lean + RZ Shotgun-lean + Pass-light | 36 | +0.45 | +0.23 | +0.41 | -0.66 | +0.42 | -0.63 |
| C3 | Pass-averse | 22 | -1.13 | -0.78 | -1.16 | -1.31 | -1.16 | -1.20 |
| C4 | No-Huddle-heavy + RZ Shotgun-heavy | 11 | +1.01 | +2.43 | +0.72 | +0.62 | +1.06 | +0.97 |

### Cluster mean raw rates

| Cluster | Label | Shotgun | No-Huddle | Deep (AY>=20) | Pass | RZ Shotgun | RZ Pass |
|---|---|---|---|---|---|---|---|
| C0 | RZ Pass-heavy + Pass-heavy | 61.9% | 7.4% | 6.6% | 60.6% | 60.5% | 56.4% |
| C1 | Pass-lean + RZ Shotgun-averse | 17.7% | 1.5% | 2.2% | 58.8% | 12.8% | 51.9% |
| C2 | Shotgun-lean + RZ Shotgun-lean + Pass-light | 55.0% | 8.1% | 5.7% | 55.7% | 51.8% | 48.7% |
| C3 | Pass-averse | 19.2% | 1.6% | 1.9% | 53.7% | 13.5% | 45.8% |
| C4 | No-Huddle-heavy + RZ Shotgun-heavy | 67.9% | 22.3% | 6.5% | 59.8% | 67.3% | 56.8% |

### Representative coaches per cluster (up to 10, by tenure)

- **C0 (RZ Pass-heavy + Pass-heavy)** (n=43): Pete Carmichael, Marty Mornhinweg, Greg Olson, Scott Linehan, Dirk Koetter, Nathaniel Hackett, Todd Haley, Eric Bieniemy, Dowell Loggains, Joe Philbin
- **C1 (Pass-lean + RZ Shotgun-averse)** (n=29): Bob Bratkowski, Tom Moore, Gil Haskell, Bruce Arians, Bill Muir, Jimmy Raye, Bill Callahan, Mike Martz, Brad Childress, Tom Rossley
- **C2 (Shotgun-lean + RZ Shotgun-lean + Pass-light)** (n=36): Darrell Bevell, Brian Schottenheimer, Josh McDaniels, Kevin Gilbride, Greg Roman, Rick Dennison, Bill Musgrave, Norv Turner, Kyle Shanahan, Brian Daboll
- **C3 (Pass-averse)** (n=22): Greg Knapp, Cam Cameron, Gary Kubiak, Mike Heimerdinger, Dan Henning, Al Saunders, Mike Mularkey, Matt Cavanaugh, Charlie Weis, Mike McCarthy
- **C4 (No-Huddle-heavy + RZ Shotgun-heavy)** (n=11): Pat Shurmur, Jim Bob Cooter, Curtis Modkins, Mike Sullivan, Frank Reich, Shane Waldron, Ben McAdoo, Edgar Bennett, Adam Gase, Bill O'Brien

## Reproduce

```bash
python "scripts/coach_fingerprint_clustering.py"
```

Outputs:
- `figures/hc_fingerprint_pca_2d.png`
- `figures/hc_fingerprint_radar.png`
- `figures/oc_fingerprint_pca_2d.png`
- `figures/oc_fingerprint_radar.png`
- `data/nfl/hc_fingerprint_clusters.csv`
- `data/nfl/oc_fingerprint_clusters.csv`