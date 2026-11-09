#!/usr/bin/env python3
"""Fig 3: peak belt dose rate versus burst location.
Renders from analysis/cache/fig3_peakmap.npz (analysis/fig3_compute.py): 1 Mt (50% fission) at 500 km,
1.289 betas per fission x <S>, IGRF-14 at 2026-01-14, 2 mm Al.
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "analysis")
sys.path.insert(0, ANALYSIS)
import hane_common  # noqa: F401  (sets the Natural Earth data path)
CACHE = os.path.join(ANALYSIS, "cache")
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.linewidth": 0.8, "savefig.dpi": 320, "savefig.bbox": "tight"})


def edges(c):
    """Cell edges; a longitude axis running -180..180 gets half-width end cells so the seam is not drawn twice."""
    dd = np.diff(c).mean(); e = np.concatenate([c - dd / 2, [c[-1] + dd / 2]])
    if np.isclose(c[0], -180) and np.isclose(c[-1], 180):
        e[0], e[-1] = -180.0, 180.0
    return e


def smooth_contour_grid(x, y, z, factor=4, passes=2):
    """Densify and lightly smooth a field for display contours only."""
    xf = np.linspace(x[0], x[-1], (len(x) - 1) * factor + 1)
    yf = np.linspace(y[0], y[-1], (len(y) - 1) * factor + 1)
    zx = np.vstack([np.interp(xf, x, row) for row in z])
    zf = np.vstack([np.interp(yf, y, zx[:, j]) for j in range(zx.shape[1])]).T
    kernel = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
    for _ in range(passes):
        zf = np.apply_along_axis(
            lambda row: np.convolve(np.pad(row, 2, mode="edge"), kernel, mode="valid"), 1, zf)
        zf = np.apply_along_axis(
            lambda col: np.convolve(np.pad(col, 2, mode="edge"), kernel, mode="valid"), 0, zf)
    return xf, yf, zf


def grat(ax):
    ax.grid(False)                                   # no latitude/longitude reference lines
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-90, 91, 45))
    ax.set_xlabel("Burst longitude [°]"); ax.set_ylabel("Burst latitude [°]")


# (a) peak belt dose-rate map
d = np.load(os.path.join(CACHE, "fig3_peakmap.npz"))
peak = np.asarray(d["peak"], dtype=float)
VMIN = 0.1
shown = np.ma.masked_where(~np.isfinite(peak) | (peak < VMIN), peak)

def sim_grid(ax, lons, lats, **kw):
    """Thin white lines on the cell edges: each cell is centred on one simulated burst location."""
    xe, ye = edges(lons), edges(lats)
    style = dict(color="white", lw=0.25, alpha=0.4, zorder=2.6)
    if kw: style["transform"] = kw["transform"]
    for x in xe:
        ax.plot([x, x], [ye[0], ye[-1]], **style)
    for y in ye:
        ax.plot([xe[0], xe[-1]], [y, y], **style)

# colour bins in log steps (rad/s)
boundaries = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
boundaries = boundaries[boundaries <= max(np.nanmax(peak) * 1.2, 1.0)]
if boundaries[-1] < np.nanmax(peak):
    boundaries = np.r_[boundaries, np.nanmax(peak) * 1.0001]   # top tick = data maximum
cmap = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(0.18, 0.92, len(boundaries) - 1)))
cmap.set_bad((0, 0, 0, 0))
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

fig, ax = plt.subplots(figsize=(9.4, 5.0), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_facecolor("white")
ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white", zorder=0)
ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d9d9d9", zorder=0)
ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="black", linewidth=0.6, zorder=1)
ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="black", linewidth=0.3, zorder=1)
im = ax.pcolormesh(edges(d["LONS"]), edges(d["LATS"]), shown,
                   cmap=cmap, norm=norm, alpha=0.75, shading="flat",
                   edgecolors="none", linewidth=0, antialiased=False, rasterized=True,
                   transform=ccrs.PlateCarree(), zorder=2)   # one seamless image (no cell outlines)
if os.environ.get("FIG3_GRID", "0") == "1":
    sim_grid(ax, d["LONS"], d["LATS"], transform=ccrs.PlateCarree())
CLEV = [l for l in np.log10([1, 10, 20, 30]) if l < np.log10(np.nanmax(peak))]
contour_lon, contour_lat, contour_peak = smooth_contour_grid(d["LONS"], d["LATS"], np.log10(np.maximum(peak, 1e-3)))
cs = ax.contour(contour_lon, contour_lat, contour_peak, levels=CLEV,
                colors="white", linewidths=0.7, alpha=0.9,
                transform=ccrs.PlateCarree(), zorder=3)
ax.clabel(cs, fmt={l: f"{10**l:.0f} rad/s" for l in CLEV}, fontsize=7.5, colors="white", inline=True)
grat(ax)
# reference burst, 20S 35W
ax.plot(-35, -20, marker="+", ms=15, mew=1.8, color="black", ls="none",
        transform=ccrs.PlateCarree(), zorder=5)
ax.plot(-35, -20, marker="o", ms=3.8, mfc="black", mec="black", ls="none",
        transform=ccrs.PlateCarree(), zorder=6)
cb = fig.colorbar(im, ax=ax, pad=0.04, fraction=0.04,
                  boundaries=boundaries,
                  ticks=boundaries,
                  spacing="uniform")
cb.ax.set_yticklabels([f"{b:g}" if b in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) else f"{b:.0f}" for b in boundaries])
cb.set_label("Peak belt dose rate [rad/s]")
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.join(HERE, f"peak_dose_rate_map.{e}"), dpi=320)
plt.close(fig)

print("Fig_3: wrote peak_dose_rate_map to", HERE)
