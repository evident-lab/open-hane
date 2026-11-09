#!/usr/bin/env python3
"""Fig 4 — cumulative LEO satellite loss: 1 Mt at 500 km, 20S 35W, 40 krad(Si) behind 2 mm Al.
  (a) fleet lost versus time: light-blue band = range between the lower (0.40 beta x <S>) and upper
      (5.21 beta x <S>) trapping cases; line = middle case (1.29 beta x <S>);
  (b) fleet status at t = 0, 1, 3 and 7 days (middle case, all LEO objects) on a 3-D Earth drawn like Fig 2
      (grey shaded globe with coastlines); light grey = operational, red = failed. Satellites behind
      the Earth are hidden.
Data: analysis/cache/fleet_saa.npz (analysis/fleet_compute.py)."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import cartopy.io.shapereader as shapereader
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "analysis")
sys.path.insert(0, ROOT)
import hane_common  # noqa: F401  (Natural Earth data path)
from style import plotter_style
from fleet_compute import loss_curve

# same look as Figs 2 and 3
plotter_style()
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
                     "axes.labelweight": "normal", "axes.titlesize": 12, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "axes.linewidth": 0.8, "axes.spines.top": True,
                     "axes.spines.right": True, "legend.fontsize": 10.5})

KRAD = 40.0
SNAP_DAYS = (0.0, 1.0, 3.0, 7.0)
INK, RED, BAND, LINE = "#222222", "#C62828", "#9ECAE1", "#163A70"
OK_FACE, OK_EDGE = "#E6E6E6", "#4A4A4A"
AZIM, ELEV = -35.0, 8.0                       # view centred on the burst side (South Atlantic)

d = np.load(os.path.join(ROOT, "cache", "fleet_saa.npz"))
t = d["times"] / 86400.0
lo, mid, hi = (loss_curve(d, c, KRAD) for c in ("low", "mid", "high"))
FK = float(d["fluence_kill"])
p0 = 100.0 * np.mean((d["Fp"] >= FK) | (d["dprompt"] >= KRAD))

fig = plt.figure(figsize=(15.0, 6.6))

# ---------------- (a) loss curves ----------------
ax = fig.add_axes([0.055, 0.13, 0.42, 0.80])
ax.fill_between(t, lo, hi, color=BAND, lw=0, label="Range: 0.40–5.21 β × ⟨S⟩")
ax.plot(t, mid, color=LINE, lw=2.4, label="1.29 β × ⟨S⟩")
ax.annotate(f"prompt X-ray\n{p0:.1f}% at t = 0", (0, p0), (4.6, 7), color=RED, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0), annotation_clip=False)
ax.set_xlim(0, 30); ax.set_ylim(0, 100)
ax.set_xlabel("Days after burst"); ax.set_ylabel("LEO satellites lost [%]")
ax.legend(loc="lower right", frameon=False)
ax.text(-0.10, 1.03, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
its = [int(np.argmin(np.abs(t - s))) for s in SNAP_DAYS]
for n, it in enumerate(its, 1):
    ax.plot(t[it], mid[it], "o", ms=16, mfc="white", mec=INK, mew=1.1, zorder=8)
    ax.text(t[it], mid[it], str(n), ha="center", va="center", fontsize=10, fontweight="bold", zorder=9)

# ---------------- (b) fleet status on the Fig 2 Earth ----------------
az, el = np.radians(AZIM), np.radians(ELEV)
view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])   # towards the viewer

# shaded grey globe, lit as in Fig 2 (light from upper left of the viewer)
u, v = np.mgrid[-np.pi:np.pi:181j, -np.pi / 2:np.pi / 2:91j]
ex, ey, ez = np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)
up = np.array([0, 0, 1.0]); right = np.cross(up, view); right /= np.linalg.norm(right)
upv = np.cross(view, right)
light = -0.35 * right + 0.45 * upv + 0.82 * view; light /= np.linalg.norm(light)
shade = 0.18 + 0.58 * np.clip(ex * light[0] + ey * light[1] + ez * light[2], 0, 1)
globe_rgb = plt.get_cmap("Greys_r")(shade)

# Natural Earth coastlines as 3-D polylines on the sphere
coast = []
for geom in shapereader.Reader(shapereader.natural_earth("110m", "physical", "coastline")).geometries():
    for part in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        c = np.radians(np.asarray(part.coords))
        coast.append(1.003 * np.c_[np.cos(c[:, 1]) * np.cos(c[:, 0]), np.cos(c[:, 1]) * np.sin(c[:, 0]), np.sin(c[:, 1])])

samp = np.arange(len(d["lat"]))                      # every LEO object (13,306)
la = np.radians(d["lat"][samp]); lon = np.radians(d["lon"][samp]); r = 1 + d["alt"][samp] / 6371.0
P = np.c_[r * np.cos(la) * np.cos(lon), r * np.cos(la) * np.sin(lon), r * np.sin(la)]
depth = P @ view
perp = np.sqrt(np.maximum((P ** 2).sum(1) - depth ** 2, 0))
visible = (depth > 0) | (perp > 1.0)                  # hide satellites behind the Earth's disk
dead_prompt = d["Fp"][samp] >= FK

pos = [[0.535, 0.50, 0.225, 0.42], [0.765, 0.50, 0.225, 0.42],
       [0.535, 0.055, 0.225, 0.42], [0.765, 0.055, 0.225, 0.42]]
for n, (it, box, sd) in enumerate(zip(its, pos, SNAP_DAYS), 1):
    a3 = fig.add_axes(box, projection="3d")
    a3.plot_surface(ex, ey, ez, facecolors=globe_rgb, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False, zorder=1)
    for seg in coast:
        front = seg @ view > 0
        xs = np.where(front, seg[:, 0], np.nan); ys = np.where(front, seg[:, 1], np.nan)
        zs = np.where(front, seg[:, 2], np.nan)
        a3.plot(xs, ys, zs, color=INK, lw=0.45, zorder=2)
    dead = dead_prompt | (d["dprompt"][samp] + d["cum_mid"][samp, it] >= KRAD)
    for mask, face, edge, zo in ((visible & ~dead, OK_FACE, OK_EDGE, 3), (visible & dead, RED, RED, 4)):
        a3.scatter(P[mask, 0], P[mask, 1], P[mask, 2], s=1.4, c=face, edgecolors=edge, linewidths=0.12,
                   depthshade=False, zorder=zo)
    a3.set_box_aspect((1, 1, 1), zoom=1.45); a3.view_init(elev=ELEV, azim=AZIM)
    a3.set_xlim(-1.3, 1.3); a3.set_ylim(-1.3, 1.3); a3.set_zlim(-1.3, 1.3); a3.set_axis_off()
    a3.computed_zorder = False
    # numbered circle as in panel (a), then the time and loss
    a3.text2D(0.20, 0.985, str(n), transform=a3.transAxes, ha="center", va="center", fontsize=10,
              fontweight="bold", color=INK, zorder=10,
              bbox=dict(boxstyle="circle,pad=0.28", facecolor="white", edgecolor=INK, linewidth=1.1))
    a3.text2D(0.27, 0.985, f"t = {sd:.0f} d  —  {mid[it]:.0f}% lost", transform=a3.transAxes,
              ha="left", va="center", fontsize=11, fontweight="bold", color=INK, zorder=10)
fig.text(0.515, 0.95, "b", fontsize=12, fontweight="bold", ha="left")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=6, mfc=OK_FACE, mec=OK_EDGE, mew=0.8, label="Operational"),
                    Line2D([], [], ls="none", marker="o", ms=6, mfc=RED, mec=RED, label="Failed")],
           loc="upper center", bbox_to_anchor=(0.763, 0.995), ncol=2, frameon=False, fontsize=11,
           handletextpad=0.3, columnspacing=1.6)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"cumulative_satellite_loss.{e}"), dpi=300)
plt.close(fig)
print("Fig_4: snapshots", [f"t={s:.0f} d: {mid[i]:.1f}%" for s, i in zip(SNAP_DAYS, its)])
