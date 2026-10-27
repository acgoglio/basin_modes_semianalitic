#!/usr/bin/env python3
"""
Automatic association between semi-analytical (SA) and numerical (BF-3)
Mediterranean basin modes, based on period proximity and spatial correlation.

For every SA mode of the comparison table:
  1. candidates = numerical groups with |T_SA - T_num| <= WIN_FRAC * T_SA
                  and coverage >= MIN_COV (same threshold as the histograms)
                  and T_num + 2*tol_num >= shortest SA period in the file
                  (shorter numerical groups cannot have an SA counterpart)
  2. for each candidate: Pearson correlation r between the two relative
     amplitude maps, computed on the sea points where both are defined
     (NaN of the numerical map excluded)
  3. candidates with r >= r_max - R_MARGIN are considered equivalent in
     spatial structure; among them the one closest in period is selected
  4. confidence from r:  High r >= R_HIGH,  Medium R_LOW <= r < R_HIGH,
     Low r < R_LOW
  5. flag (no change of confidence) if |T_SA - T_num| > 2 * tol_num,
     i.e. outside the resolution of the numerical procedure

For comparison, r is also computed for the association chosen by eye
(column T_num_manual), so that the automatic and manual results can be
compared row by row.

Output: OUTCSV (one row per SA mode), read by the plotting script;
        OUTCSV_CAND (one row per SA mode / candidate pair), diagnostic.
Python 3.7 compatible.
"""

import os
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
SA_FILE  = "/work/cmcc/ag15419/basin_modes_new/basin_modes_sa_sign_comb/med_modes_full_120_T3.nc"
NUM_DIR  = "/work/cmcc/ag15419/basin_modes_new/basin_modes_1_fg/area_20_newFFT_ok_bis_opt/"
NUM_FILE = os.path.join(NUM_DIR, "mode_groups_amp.nc")
NUM_CSV  = os.path.join(NUM_DIR, "periods_grouped_amp.csv")
OUTCSV   = "sa_num_match.csv"
OUTCSV_CAND = "sa_num_candidates.csv"

WIN_FRAC = 0.25      # candidate window: |dT| <= WIN_FRAC * T_SA
MIN_COV  = 7.0       # [%] minimum coverage of a numerical group to be a candidate
R_MARGIN = 0.1      # candidates with r >= r_max - R_MARGIN are "tied" in structure
R_HIGH   = 0.9
R_LOW    = 0.7
R_DEC    = 2        # decimals of r: used for the class and written in the CSVs
SA_PER_TOL = 0.005   # [h] tolerance to find the table SA periods in the file
MIN_PTS  = 100       # minimum number of common points to compute r

# Mediterranean sea mask (same definition as mode_period_plot_ampval.py)
LON_MIN, LON_MAX = -6.0, 36.3
LAT_MIN, LAT_MAX = 30.0, 46.0

# Relative amplitude, as in the maps: A / p99, clipped to 1.
# NOTE: Pearson r does not depend on the normalisation; it only changes
# with the clipping (values above p99 set to 1). Set CLIP_P99 = False to
# use the unclipped amplitudes.
CLIP_P99 = True

# Sea points where a numerical group was not detected (NaN in the map):
#   False -> excluded from the correlation (common points only)
#   True  -> set to relative amplitude 0 ("mode absent there"), so every
#            candidate is compared with the SA map over the whole sea domain
NAN_AS_ZERO = True
if NAN_AS_ZERO:   # separate output files, so the two runs can be compared
    OUTCSV = OUTCSV.replace(".csv", "_nan0.csv")
    OUTCSV_CAND = OUTCSV_CAND.replace(".csv", "_nan0.csv")

# --------------------------------------------------------------------------- #
# comparison table: (area, T_SA, T_num chosen by eye, confidence by eye)
# (3.97 h of the split group is not used here)
# --------------------------------------------------------------------------- #
TABLE = [
    ("Mediterranean Basin",          26.65, 27.00, "High"),
    ("Adriatic Sea",                 21.22, 21.00, "High"),
    ("Aegean Sea / Gulf of Gabes",   13.32, 13.80, "High"),
    ("Gulf of Gabes / Aegean Sea",   11.49, 11.90, "High"),
    ("Adriatic Sea / Gulf of Gabes", 10.73, 10.90, "High"),
    ("Adriatic Sea / Gulf of Gabes",  9.37,  9.50, "High"),
    ("Tyrrhenian Sea / Alboran Sea",  8.41,  8.50, "High"),
    ("Adriatic Sea / Gulf of Gabes",  7.08,  7.20, "High"),
    ("Gulf of Gabes / Adriatic Sea",  6.71,  6.90, "High"),
    ("Alboran Sea",                   5.96,  5.75, "High"),
    ("Gulf of Gabes",                 5.70,  5.99, "Medium"),
    ("Adriatic Sea",                  5.64,  5.99, "Medium"),
    ("Sicily Strait",                 5.26,  6.40, "Low"),
    ("Adriatic Sea",                  5.11,  5.47, "Medium"),
    ("Aegean Sea",                    4.98,  5.22, "High"),
    ("Adriatic Sea",                  4.91,  5.47, "Medium"),
    ("Aegean Sea",                    4.25,  4.34, "High"),
    ("Gulf of Gabes / Aegean Sea",    4.20,  4.20, "High"),
    ("Gulf of Gabes",                 4.09,  4.03, "Medium"),
    ("Aegean Sea",                    3.86,  3.85, "Low"),
    ("Gulf of Gabes",                 3.85,  3.85, "Low"),
    ("Adriatic Sea",                  3.82,  3.85, "Low"),
    ("Gulf of Gabes",                 3.72,  3.68, "Medium"),
    ("Nile Delta",                    3.70,  3.68, "Medium"),
    ("Aegean Sea",                    3.67,  3.68, "Medium"),
    ("Gulf of Gabes",                 3.66,  3.68, "Medium"),
    ("West Mediterranean",            3.63,  3.63, "High"),
    ("Adriatic Sea",                  3.52,  3.58, "Low"),
    ("Aegean Sea",                    3.47,  3.58, "Low"),
    ("Aegean Sea",                    3.44,  3.58, "Low"),
    ("Gulf of Gabes",                 3.41,  3.58, "Low"),
    ("East Mediterranean",            3.35,  3.46, "High"),
    ("Adriatic Sea",                  3.32,  3.58, "Low"),
    ("Lybian Gulf",                   3.29,  3.28, "Low"),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rel_amp(field, valid):
    """relative amplitude |field| / p99 on the valid points (NaN elsewhere)"""
    a = np.abs(np.asarray(field, dtype=float))
    out = np.full(a.shape, np.nan)
    p99 = np.nanpercentile(a[valid], 99)
    if not np.isfinite(p99) or p99 <= 0:
        return out
    out[valid] = a[valid] / p99
    if CLIP_P99:
        out = np.clip(out, 0.0, 1.0)
    return out


def pearson(a, b, valid):
    """Pearson correlation between a and b on the points where both are finite"""
    m = valid & np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < MIN_PTS:
        return np.nan, n
    x, y = a[m], b[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, n
    return float(np.corrcoef(x, y)[0, 1]), n


def conf_from_r(r):
    if not np.isfinite(r):
        return "None"
    if r >= R_HIGH:
        return "High"
    if r >= R_LOW:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# read numerical groups
# --------------------------------------------------------------------------- #
df_grp = pd.read_csv(NUM_CSV)
for col in ("Grouped_Period", "Tolerance", "Percentage"):
    if col not in df_grp.columns:
        sys.exit(f"ERROR: column {col} not found in {NUM_CSV}")

ds_num = nc.Dataset(NUM_FILE)
nav_lon = np.asarray(ds_num.variables["nav_lon"][:])
nav_lat = np.asarray(ds_num.variables["nav_lat"][:])

groups = []   # list of dict: T, tol, cov, rel (relative amplitude map)
missing = []
for _, row in df_grp.iterrows():
    gp = float(row["Grouped_Period"])
    vname = f"amp_{gp:.2f}h"
    if vname not in ds_num.variables:
        missing.append(vname)
        continue
    fld = np.ma.filled(ds_num.variables[vname][:].astype(float), np.nan)
    groups.append({"T": gp, "tol": float(row["Tolerance"]),
                   "cov": float(row["Percentage"]), "raw": fld})
ds_num.close()
if missing:
    print(f"WARNING: {len(missing)} groups of the CSV not found in the netCDF: {missing}")
print(f"numerical groups read: {len(groups)}  "
      f"(coverage >= {MIN_COV}%: {sum(g['cov'] >= MIN_COV for g in groups)})")

# --------------------------------------------------------------------------- #
# read SA modes
# --------------------------------------------------------------------------- #
ds_sa = nc.Dataset(SA_FILE)
sa_modes = ds_sa.variables["modes"]
sa_per = np.asarray(ds_sa.variables["periods"][:], dtype=float)
T_SA_MIN_FILE = float(np.nanmin(sa_per))
n_short = sum(g["T"] + 2.0 * g["tol"] < T_SA_MIN_FILE for g in groups)
print(f"shortest SA period in the file: {T_SA_MIN_FILE:.3f} h -> "
      f"{n_short} numerical groups excluded (T_num + 2*tol < this value)")
sa_mask = (np.asarray(ds_sa.variables["mask"][:]) == 1) if "mask" in ds_sa.variables \
    else np.ones(nav_lon.shape, dtype=bool)

if sa_modes.shape[1:] != nav_lon.shape:
    sys.exit(f"ERROR: SA grid {sa_modes.shape[1:]} != numerical grid {nav_lon.shape}")

# common sea mask
box = (nav_lon >= LON_MIN) & (nav_lon <= LON_MAX) & \
      (nav_lat >= LAT_MIN) & (nav_lat <= LAT_MAX)
atlantic = (nav_lon < 0) & (nav_lat > 42)
sea = sa_mask & box & (~atlantic)
print(f"common sea points: {int(sea.sum())}")
print(f"NAN_AS_ZERO = {NAN_AS_ZERO}")

# relative amplitude maps of the numerical groups (on the sea points)
for g in groups:
    g["rel"] = rel_amp(g["raw"], sea & np.isfinite(g["raw"]))
    if NAN_AS_ZERO:
        g["rel"][sea & ~np.isfinite(g["raw"])] = 0.0
    del g["raw"]

# --------------------------------------------------------------------------- #
# association
# --------------------------------------------------------------------------- #
rows = []
cand_rows = []
for k, (area, t_sa, t_man, conf_man) in enumerate(TABLE):

    # SA mode in the file
    hit = np.where(np.abs(sa_per - t_sa) <= SA_PER_TOL)[0]
    if len(hit) != 1:
        sys.exit(f"ERROR: SA period {t_sa} h found {len(hit)} times in {SA_FILE} "
                 f"(periods within 0.05 h: {sa_per[np.abs(sa_per - t_sa) < 0.05]})")
    sa_fld = np.ma.filled(sa_modes[hit[0]].astype(float), np.nan)
    sa_rel = rel_amp(sa_fld, sea & np.isfinite(sa_fld))

    # candidates
    cands = [g for g in groups
             if abs(g["T"] - t_sa) <= WIN_FRAC * t_sa and g["cov"] >= MIN_COV
             and g["T"] + 2.0 * g["tol"] >= T_SA_MIN_FILE]
    scores = []
    for g in cands:
        r, n = pearson(sa_rel, g["rel"], sea)
        scores.append((r, n, g))
    scores_ok = [s for s in scores if np.isfinite(s[0])]
    scores_ok.sort(key=lambda s: s[0], reverse=True)

    # correlation with the association chosen by eye (closest group to t_man)
    g_man = min(groups, key=lambda g: abs(g["T"] - t_man))
    r_man, n_man = pearson(sa_rel, g_man["rel"], sea)

    # all candidates, for diagnostics
    for rr, nn, gg in scores:
        cand_rows.append(dict(idx=k, T_SA=t_sa, T_num=gg["T"], tol_num=gg["tol"],
                              cov_num=gg["cov"], dT=gg["T"] - t_sa,
                              dT_over_tol=(gg["T"] - t_sa) / gg["tol"],
                              r=rr, n_pts=nn))

    if scores_ok:
        r_max, _, g_rmax = scores_ok[0]
        tied = [s_ for s_ in scores_ok if s_[0] >= r_max - R_MARGIN]
        r, n, g = min(tied, key=lambda s_: abs(s_[2]["T"] - t_sa))
        others = [s_ for s_ in scores_ok if s_[2] is not g]
        r2 = others[0][0] if others else np.nan
        T2 = others[0][2]["T"] if others else np.nan
        dT = g["T"] - t_sa
        out_res = abs(dT) > 2.0 * g["tol"]
        rows.append(dict(
            idx=k, area=area, T_SA=t_sa, SA_file_index=int(hit[0]),
            n_cand=len(cands), n_tied=len(tied),
            T_num=g["T"], tol_num=g["tol"], cov_num=g["cov"],
            dT=dT, dT_over_tol=dT / g["tol"], outside_2tol=out_res,
            r=r, n_pts=n, conf_auto=conf_from_r(round(r, R_DEC)),
            T_num_rmax=g_rmax["T"], r_max=r_max,
            T_num_2nd=T2, r_2nd=r2,
            T_num_manual=g_man["T"], r_manual=r_man, conf_manual=conf_man,
            same_as_manual=bool(np.isclose(g["T"], g_man["T"]))))
    else:
        rows.append(dict(
            idx=k, area=area, T_SA=t_sa, SA_file_index=int(hit[0]),
            n_cand=len(cands), n_tied=0,
            T_num=np.nan, tol_num=np.nan, cov_num=np.nan,
            dT=np.nan, dT_over_tol=np.nan, outside_2tol=False,
            r=np.nan, n_pts=0, conf_auto="None",
            T_num_rmax=np.nan, r_max=np.nan,
            T_num_2nd=np.nan, r_2nd=np.nan,
            T_num_manual=g_man["T"], r_manual=r_man, conf_manual=conf_man,
            same_as_manual=False))
ds_sa.close()

df = pd.DataFrame(rows)
# r-type columns rounded to R_DEC decimals (same value used for the class);
# all other columns keep 3 decimals
R_COLS = ["r", "r_max", "r_2nd", "r_manual"]
df[R_COLS] = df[R_COLS].round(R_DEC)
df.to_csv(OUTCSV, index=False, float_format="%.3f")
print(f"saved: {OUTCSV}")
df_cand = pd.DataFrame(cand_rows).sort_values(["idx", "r"], ascending=[True, False])
df_cand["r"] = df_cand["r"].round(R_DEC)
df_cand.to_csv(OUTCSV_CAND, index=False, float_format="%.3f")
print(f"saved: {OUTCSV_CAND}")

# --------------------------------------------------------------------------- #
# summary on screen
# --------------------------------------------------------------------------- #
pd.set_option("display.width", 200)
print(df[["idx", "T_SA", "T_num", "dT_over_tol", "outside_2tol", "r", "n_tied", "T_num_rmax",
          "conf_auto", "T_num_manual", "r_manual", "conf_manual"]]
      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\nsame association as by eye: {int(df['same_as_manual'].sum())}/{len(df)}")
print("confidence auto vs manual (rows: manual, columns: auto):")
print(pd.crosstab(df["conf_manual"], df["conf_auto"]))
