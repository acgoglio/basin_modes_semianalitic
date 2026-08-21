#!/usr/bin/env python3
"""
Comparison of the Mediterranean oscillatory basin modes obtained with the
semi-analytical procedure and with the numerical (BF-3) free-oscillation
experiment.

Upper panel
    x : mode index (semi-analytical modes sorted by decreasing period)
    y : mode period [h]
        - semi-analytical results  -> markers
        - numerical results        -> bars spanning  value +/- uncertainty

Lower panel  (resolvability of the numerical procedure)
    x : mode index
    y : hours
        - spacing between consecutive semi-analytical modes,  dT_SA
        - width of the grouping window of the numerical procedure,  2*dT_num
    Where dT_SA < 2*dT_num two neighbouring semi-analytical modes fall inside
    the same numerical grouping window and cannot be resolved separately,
    irrespective of how the association is performed.

Values are hard-coded from the comparison tables.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
mpl.use('Agg')

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
LOG_Y = True             # log scale on the period axis (periods span 3-27 h)
SHOW_SPLIT = False       # also show the secondary numerical mode of split groups
SHADE_UNRESOLVED = True  # highlight where dT_SA < 2*dT_num in the lower panel
SHOW_LITERATURE = True   # published periods obtained including rotation
SHOW_F0 = False           # also show the no-rotation (f=0) results, in light grey

# The four series are dodged horizontally inside each bar so that they do not
# overlap. The x axis is a discrete mode index, so the offsets carry no
# meaning: only the vertical position (the period) is quantitative.
# the semi-analytical (f != 0) mode is NOT dodged: it stays centred on the
# numerical bar it is compared with. The other three series are dodged.
DX_SA, DX_SA0, DX_LIT, DX_LIT0 = 0.0, -0.17, 0.15, 0.29
MS_SA, MS_LIT = 5.5, 6.5    # marker sizes
C_F0 = "0.62"               # light grey used for the f=0 results
OUTFILE = "modes_comparison_2panels.png"

# colours by match confidence
CONF_COLOR = {"High": "#2166ac", "Medium": "#f4a300", "Low": "#b2182b"}

# --------------------------------------------------------------------------- #
# data:  (area, T_semianalytic, T_numerical, unc_numerical, confidence)
# T_numerical = None  ->  no numerical counterpart
# modes are listed by strictly decreasing semi-analytical period, so that the
# index matches the mode numbering used in the figures
# --------------------------------------------------------------------------- #
MODES = [
    ("Mediterranean Basin",        26.65, 27.00, 2.00,  "High"),
    ("Adriatic Sea",               21.22, 21.00, 1.00,  "High"),
    ("Aegean Sea / Gulf of Gabes", 13.32, 13.80, 0.40,  "High"),
    ("Gulf of Gabes / Aegean Sea", 11.49, 11.90, 0.30,  "High"),
    ("Adriatic Sea / Gulf of Gabes", 10.73, 10.90, 0.20, "High"),
    ("Adriatic Sea / Gulf of Gabes", 9.37,  9.50, 0.20,  "High"),
    ("Tyrrhenian Sea / Alboran Sea", 8.41,  8.50, 0.20,  "High"),
    ("Adriatic Sea / Gulf of Gabes", 7.08,  7.20, 0.10,  "High"),
    ("Gulf of Gabes / Adriatic Sea", 6.71,  6.90, 0.10,  "High"),
    ("Alboran Sea",                 5.96,  5.75, 0.07,  "High"),
    ("Gulf of Gabes",               5.70,  5.99, 0.08,  "Medium"),
    ("Adriatic Sea",                5.64,  5.99, 0.08,  "Medium"),
    ("Sicily Strait",               5.26,  6.40, 0.09,  "Low"),
    ("Adriatic Sea",                5.11,  5.47, 0.06,  "Medium"),
    ("Aegean Sea",                  4.98,  5.22, 0.06,  "High"),
    ("Adriatic Sea",                4.91,  5.47, 0.06,  "Medium"),
    ("Aegean Sea",                  4.25,  4.34, 0.04,  "High"),
    ("Gulf of Gabes / Aegean Sea",  4.20,  4.20, 0.04,  "High"),
    ("Gulf of Gabes",               4.09,  4.03, 0.03,  "Medium"),
    ("Aegean Sea",                  3.86,  3.85, 0.03,  "Low"),
    ("Gulf of Gabes",               3.85,  3.85, 0.03,  "Low"),
    ("Adriatic Sea",                3.82,  3.85, 0.03,  "Low"),
    ("Gulf of Gabes",               3.72,  3.68, 0.03,  "Medium"),
    ("Nile Delta",                  3.70,  3.68, 0.03,  "Medium"),
    ("Aegean Sea",                  3.67,  3.68, 0.03,  "Medium"),
    ("Gulf of Gabes",               3.66,  3.68, 0.03,  "Medium"),
    ("West Mediterranean",          3.63,  3.63, 0.03,  "High"),
    ("Adriatic Sea",                3.52,  3.58, 0.03,  "Low"),
    ("Aegean Sea",                  3.47,  3.58, 0.03,  "Low"),
    ("Aegean Sea",                  3.44,  3.58, 0.03,  "Low"),
    ("Gulf of Gabes",               3.41,  3.58, 0.03,  "Low"),
    ("East Mediterranean",          3.35,  3.46, 0.03,  "High"),
    ("Adriatic Sea",                3.32,  3.58, 0.03,  "Low"),
    ("Lybian Gulf",                 3.29,  3.28, 0.02,  "Low"),
]

# groups associated with two numerical modes: {mode index: (T, unc)}
SPLIT = {19: (3.97, 0.03), 20: (3.97, 0.03), 21: (3.97, 0.03)}

# --------------------------------------------------------------------------- #
# published periods obtained INCLUDING the Earth's rotation (f != 0), listed
# against the semi-analytical mode they are compared with.
# NOTE: 11.4 h (Sternec) and 9.9 h (Sguazzero) have a provisional assignment
# in the tables: the treatment of rotation could not be verified on the
# original source. Drop them here if you prefer to plot only confirmed values.
# --------------------------------------------------------------------------- #
LITERATURE = {
    0:  [31.2, 38.5],
    1:  [18.6, 20.1, 21.1, 21.1, 21.3, 21.5, 22.0, 23.0, 23.0],
    2:  [12.03],
    3:  [11.4, 12.8],
    4:  [9.3, 9.9, 10.7, 10.9, 11.1, 12.0],
    5:  [9.3],
    6:  [8.3, 8.3, 8.4],
    7:  [7.4],
    8:  [6.8],
    9:  [6.2, 6.25, 6.96],
    11: [5.3],
    12: [5.3],
}

# semi-analytical periods computed without rotation (f = 0), same order as MODES
T_SA_F0 = [
    35.68, 22.07, 13.58, 11.63, 10.69, 9.40, 8.45, 7.05, 6.69, 5.96,
    5.74, 5.64, 5.31, 5.12, 4.98, 4.94, 4.24, 4.19, 4.06, 3.85,
    3.84, 3.82, 3.74, 3.72, 3.70, 3.68, 3.61, 3.53, 3.47, 3.44,
    3.41, 3.35, 3.32, 3.29,
]

# published periods obtained WITHOUT rotation (f = 0)
LITERATURE_F0 = {
    0: [42.5],
    1: [19.2, 20.0, 21.0, 21.9],
    4: [9.3, 10.7, 12.5],
    5: [9.3],
    6: [8.4],
    7: [7.3],
    8: [6.7, 6.8],
}

# index of the last mode with T > 6 h (vertical separator drawn at +0.5)
I_SPLIT = 9

# --------------------------------------------------------------------------- #
# derived quantities
# --------------------------------------------------------------------------- #
idx = np.arange(len(MODES))
t_sa = np.array([m[1] for m in MODES])
unc = np.array([m[3] for m in MODES])

# spacing between consecutive semi-analytical modes (NaN for the last one)
dt_sa = np.full(len(MODES), np.nan)
dt_sa[:-1] = t_sa[:-1] - t_sa[1:]

# width of the grouping window of the numerical procedure
win = 2.0 * unc

# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(13, 10), sharex=True,
    gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08})

# =========================== upper panel =================================== #
for i, (_, _, t_num, u, conf) in enumerate(MODES):
    if t_num is None:
        continue
    ax.bar(i, height=2 * u, bottom=t_num - u, width=0.72,
           color=CONF_COLOR[conf], alpha=0.45,
           edgecolor=CONF_COLOR[conf], linewidth=0.8, zorder=1.5)
    ax.hlines(t_num, i - 0.36, i + 0.36,
              color=CONF_COLOR[conf], linewidth=1.4, zorder=2)

if SHOW_SPLIT:
    for i, (t_num, u) in SPLIT.items():
        conf = MODES[i][4]
        ax.bar(i, height=2 * u, bottom=t_num - u, width=0.72,
               facecolor="none", edgecolor=CONF_COLOR[conf],
               linewidth=0.9, linestyle="--", zorder=2)

def _scatter(d, dx):
    """flatten {index: [values]} into paired x, y lists, applying the offset"""
    return (np.array([i for i, vals in d.items() for _ in vals], dtype=float) + dx,
            [v for vals in d.values() for v in vals])

# semi-analytical modes (f != 0)
ax.plot(idx + DX_SA, t_sa, linestyle="none", marker="o", markersize=MS_SA,
        markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.2, zorder=6)

# semi-analytical modes without rotation
if SHOW_F0:
    ax.plot(idx + DX_SA0, T_SA_F0, linestyle="none", marker="o",
            markersize=MS_SA, markerfacecolor="white", markeredgecolor=C_F0,
            markeredgewidth=1.2, zorder=5)

# published values obtained including rotation
if SHOW_LITERATURE:
    lx, ly = _scatter(LITERATURE, DX_LIT)
    ax.plot(lx, ly, linestyle="none", marker="x", markersize=MS_LIT,
            markeredgecolor="black", markeredgewidth=1.3, zorder=4)

# published values obtained without rotation
if SHOW_LITERATURE and SHOW_F0:
    lx, ly = _scatter(LITERATURE_F0, DX_LIT0)
    ax.plot(lx, ly, linestyle="none", marker="x", markersize=MS_LIT,
            markeredgecolor=C_F0, markeredgewidth=1.3, zorder=3)

if LOG_Y:
    ax.set_yscale("log")
    ticks = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()

ax.set_ylabel("Period (hours)", fontsize=16)
ax.set_title("Mediterranean oscillatory basin modes: "
             "semi-analytical vs numerical results", fontsize=18)
ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)

ax.axvline(I_SPLIT + 0.5, color="black", linewidth=1.2, alpha=0.7)
ax.text(I_SPLIT + 0.35, ax.get_ylim()[1] - 0.5, " T > 6 h ",
        ha="right", va="top", fontsize=12)
ax.text(I_SPLIT + 0.65, ax.get_ylim()[1] - 0.5, " T < 6 h ",
        ha="left", va="top", fontsize=12)

handles = [
    Line2D([], [], linestyle="none", marker="o", markersize=6,
           markerfacecolor="white", markeredgecolor="black",
           label="Semi-analytic mode"),
]
if SHOW_F0:
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6,
                          markerfacecolor="white", markeredgecolor=C_F0,
                          label=r"Semi-analytic mode ($f = 0$)"))
if SHOW_LITERATURE:
    handles.append(Line2D([], [], linestyle="none", marker="x", markersize=6,
                          markeredgecolor="black", markeredgewidth=1.3,
                          label=r"Literature ($f \neq 0$)"))
if SHOW_LITERATURE and SHOW_F0:
    handles.append(Line2D([], [], linestyle="none", marker="x", markersize=6,
                          markeredgecolor=C_F0, markeredgewidth=1.3,
                          label=r"Literature ($f = 0$)"))
if SHOW_SPLIT:
    handles.append(Patch(facecolor="none", edgecolor="grey", linestyle="--",
                         label="Numerical mode (split group)"))
handles += [Patch(facecolor=c, alpha=0.45, edgecolor=c,
                  label=fr"Numerical mode ($T \pm \Delta T$) $-$ {k.lower()} match confidence")
            for k, c in CONF_COLOR.items()]
ax.legend(handles=handles, loc="upper right", fontsize=14, framealpha=0.95)

# =========================== lower panel =================================== #
# shading where neighbouring semi-analytical modes cannot be resolved
if SHADE_UNRESOLVED:
    unres = dt_sa < win
    for i in idx[:-1]:
        if unres[i]:
            ax2.axvspan(i - 0.5, i + 0.5, color="#b2182b", alpha=0.10, zorder=0)

ax2.plot(idx, dt_sa, marker="o", markersize=5, linewidth=1.4,
         color="black", markerfacecolor="white", markeredgewidth=1.2,
         label=r"$\Delta T_{\rm SA}$  (spacing between consecutive"
               r" semi-analytic modes)", zorder=3)
ax2.plot(idx, win, marker="s", markersize=4.5, linewidth=1.4,
         color="#b2182b", label=r"$2\,\Delta T$  (grouping window"
                                r" of the numerical procedure)", zorder=3)

ax2.set_yscale("log")
yt = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
ax2.set_yticks(yt)
ax2.set_yticklabels([str(t) for t in yt])
ax2.minorticks_off()

ax2.set_xlabel("Mode index", fontsize=16)
ax2.set_ylabel("Hours", fontsize=16)
ax2.set_xticks(idx)
ax2.set_xticklabels(idx, fontsize=12)
ax2.tick_params(axis="y", labelsize=12)
ax2.set_xlim(-0.8, len(MODES) - 0.2)
ax2.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
ax2.set_axisbelow(True)
ax2.axvline(I_SPLIT + 0.5, color="black", linewidth=1.2, alpha=0.7)

h2, l2 = ax2.get_legend_handles_labels()
if SHADE_UNRESOLVED:
    h2.append(Patch(facecolor="#b2182b", alpha=0.10,
                    label="$\\Delta T_{\\rm SA} < 2\\,\\Delta T$: modes unresolvable through numerical procedure"))
ax2.legend(handles=h2, loc="upper right", fontsize=12, framealpha=0.95)

fig.savefig(OUTFILE, dpi=300, bbox_inches="tight")
print(f"saved: {OUTFILE}")
