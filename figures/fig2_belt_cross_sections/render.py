#!/usr/bin/env python3
"""Figure 2 — artificial-belt overlays: a mid-latitude burst and the most intense burst (20S 35W).

Belt intensity is the peak dose rate behind 2 mm Al for the middle trapping case, 1.289 betas per
fission x <S> (all debris in the atmosphere by 20 s), 1 Mt with 50% fission, IGRF-14 at 2026-01-14.
For this instantaneous injection the dose rate is largest at t = 0, so the peak is the initial value.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Wedge
import numpy as np
import cartopy.io.shapereader as shapereader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "analysis"))
import hane_common as F
from lgrid import _bmin_batch
from hanetrap import core as C
from style import plotter_style

RE = 6371.0
TILT_DEG = 11.5
LIM = 3.6
INK = "#222222"
GRAY = "#555555"
REF = "#8A8A8A"
RED = "#8B1E1E"
LEO_COLOR = "#9ecae1"

CASES = (
    dict(key="dipole", burst_lat_deg=40.75, burst_lon_deg=-35.5,
         burst_alt_km=485.0, location="40.75°N, 35.5°W", altitude="485 km"),
    dict(key="saa", burst_lat_deg=-20.0, burst_lon_deg=-35.0,
         burst_alt_km=500.0, location="20°S, 35°W", altitude="500 km"),
)


def draw_orthographic_lines(ax, category, name, center_lon, center_lat,
                            linewidth, alpha=1.0):
    """Project Natural Earth linework onto the unit Earth disk."""
    path = shapereader.natural_earth(
        resolution="110m", category=category, name=name)
    lon0, lat0 = np.radians([center_lon, center_lat])
    for geometry in shapereader.Reader(path).geometries():
        parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
        for part in parts:
            coords = np.asarray(part.coords)
            lon = np.radians(coords[:, 0]); lat = np.radians(coords[:, 1])
            dlon = lon - lon0
            visible = (np.sin(lat0) * np.sin(lat)
                       + np.cos(lat0) * np.cos(lat) * np.cos(dlon)) >= 0
            x = np.cos(lat) * np.sin(dlon)
            y = (np.cos(lat0) * np.sin(lat)
                 - np.sin(lat0) * np.cos(lat) * np.cos(dlon))
            ax.plot(np.where(visible, x, np.nan), np.where(visible, y, np.nan),
                    color=INK, lw=linewidth, alpha=alpha, zorder=3.5)


def burst_L(lat, lon, alt):
    """L of the burst point, L = (31,200 nT / B_min)^(1/3)."""
    T = F.table()
    P = np.array([C.geodetic_to_ecef(np.radians(lat), np.radians(lon), alt)])
    bm = np.zeros(1); _bmin_batch(P, T.G, T.H, T.nmax, bm)
    return (F.B0_OH / bm[0]) ** (1 / 3)


def compute_case(case):
    cells = F.region_cells(case["burst_lat_deg"], case["burst_lon_deg"], case["burst_alt_km"], n_cells=1536)
    intensity = F.dose_rate_initial(F.L_weights(cells, smooth=0.005), F.BETAS["mid"])
    Lb = burst_L(case["burst_lat_deg"], case["burst_lon_deg"], case["burst_alt_km"])
    rb = (RE + case["burst_alt_km"]) / RE
    coslam2 = min(1.0, rb / float(Lb))
    mlat = np.sign(case["burst_lat_deg"] or 1.0) * np.arccos(np.sqrt(coslam2))
    print(f"{case['key']}: <S> = {cells['meanS']:.4f}, shells L {np.nanmin(cells['L']):.3f}-"
          f"{np.nanmax(cells['L']):.3f}, burst L {Lb:.3f}, peak {intensity.max():.2f} rad/s")
    Lf = cells["L"][np.isfinite(cells["L"])]
    Ltrap = cells["L"][cells["S"] > 0]
    return dict(case=case, L=F.LG, intensity=intensity, region=dict(L_burst=Lb),
                mlat=mlat, vmax=float(intensity.max()), meanS=cells["meanS"],
                Ldeb=(float(Lf.min()), float(Lf.max())),
                Ltrap_min=float(Ltrap.min()) if len(Ltrap) else np.nan)


def draw_panel(ax, result, norm):
    case = result["case"]
    L_grid = result["L"]
    intensity = result["intensity"]
    region = result["region"]
    mlat = result["mlat"]
    lam = np.linspace(-np.pi / 2, np.pi / 2, 2000)

    r_in, r_out = 1 + 200 / RE, 1 + 2000 / RE
    ax.add_patch(Wedge((0, 0), r_out, 0, 360, width=r_out - r_in,
                       facecolor=LEO_COLOR, alpha=0.55, edgecolor="none", zorder=2.2))
    if case["key"] == "dipole":
        shell_radius = 1 + 550 / RE
        shell_angle = np.radians(145)
        ax.annotate("Mega-constellation shells\n(~500–600 km)",
                    xy=(shell_radius * np.cos(shell_angle),
                        shell_radius * np.sin(shell_angle)),
                    xytext=(-3.45, 1.9), color="#4F88A8", fontsize=7.5,
                    ha="left", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
                    arrowprops=dict(arrowstyle="->", color="#4F88A8", lw=0.9),
                    zorder=6)

    rc = (RE + case["burst_alt_km"]) / RE
    radius = 2410 / RE
    cx, cz = rc * np.cos(mlat), rc * np.sin(mlat)
    ax.add_patch(Circle((cx, cz), radius, facecolor=RED, alpha=0.20,
                        edgecolor=RED, lw=1.2, zorder=2.5))
    if case["key"] == "dipole":
        ax.annotate("Prompt-effects range\n(~2,400-km radius)",
                    xy=(cx + 0.70 * radius, cz + 0.45 * radius),
                    xytext=(1.35, 1.55), color=RED, fontsize=7.5,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.9), zorder=6)

    nn = 650
    xx = np.linspace(-LIM, LIM, nn)
    zz = np.linspace(-2.2, 2.2, int(nn * 2.2 / LIM))
    X, Z = np.meshgrid(xx, zz)
    R = np.sqrt(X ** 2 + Z ** 2)
    latitude = np.arctan2(Z, np.abs(X))
    field_L = R / np.cos(latitude) ** 2
    image = np.interp(field_L, L_grid, intensity, left=0, right=0)
    image[R < 1.0] = np.nan
    image[image < norm.vmin] = np.nan
    mesh = ax.contourf(
        X, Z, np.ma.masked_invalid(image),
        levels=np.geomspace(norm.vmin, norm.vmax, 192),
        cmap="cividis", norm=norm, antialiased=True, zorder=2.35)

    # Shaded grayscale Earth sphere (display only; geometry remains a unit disk).
    sphere_axis = np.linspace(-1, 1, 360)
    sphere_x, sphere_z = np.meshgrid(sphere_axis, sphere_axis)
    sphere_r2 = sphere_x ** 2 + sphere_z ** 2
    sphere_y = np.sqrt(np.clip(1.0 - sphere_r2, 0.0, None))
    light = np.array([-0.35, 0.45, 0.82]); light /= np.linalg.norm(light)
    shade = 0.18 + 0.58 * np.clip(
        sphere_x * light[0] + sphere_z * light[1] + sphere_y * light[2], 0, 1)
    shade[sphere_r2 > 1.0] = np.nan
    ax.imshow(np.ma.masked_invalid(shade), extent=(-1, 1, -1, 1), origin="lower",
              cmap="Greys_r", vmin=0, vmax=1, interpolation="bilinear", zorder=3)
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="white",
                        linewidth=0.8, zorder=3.1))
    draw_orthographic_lines(
        ax, "physical", "coastline",
        case["burst_lon_deg"] - 90.0, 0.0, linewidth=0.50)
    draw_orthographic_lines(
        ax, "cultural", "admin_0_boundary_lines_land",
        case["burst_lon_deg"] - 90.0, 0.0, linewidth=0.25, alpha=0.75)
    for shell in (1, 2, 3):
        radius_shell = shell * np.cos(lam) ** 2
        outside = radius_shell >= 1 - 1e-9
        for sign in (1, -1):
            x = sign * radius_shell * np.cos(lam)
            z = radius_shell * np.sin(lam)
            ax.plot(np.where(outside, x, np.nan), np.where(outside, z, np.nan),
                    color=INK, lw=1.2, zorder=4)
        if shell > 1:
            ax.text(shell + 0.05, 0.06, f"L = {shell}", color=INK,
                    fontsize=9, fontweight="normal", zorder=5)

    ax.plot(cx, cz, marker="+", ms=12, mew=1.6, color=RED, ls="none", zorder=7)
    ax.plot(cx, cz, marker="o", ms=3.5, mfc=RED, mec=RED, ls="none", zorder=7.1)
    ax.plot([0, 0], [-2.1, 2.1], color=REF, lw=1.0, ls="--", zorder=2)
    tilt = np.radians(TILT_DEG)
    tipx, tipz = 2.1 * np.sin(tilt), 2.1 * np.cos(tilt)
    ax.plot([-tipx, tipx], [-tipz, tipz], color=INK, lw=1.2, zorder=2)
    magnetic_label = "Magnetic axis"
    spin_label = "Spin axis"
    ax.text(-0.15, 2.30, magnetic_label, color=GRAY,
            fontsize=7.5, ha="right", va="top")
    ax.text(tipx + 0.12, tipz, spin_label,
            color=INK, fontsize=7.5, ha="left", va="center")

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-2.2, 2.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(-2, 3))
    ax.set_xlabel(r"Distance in magnetic equatorial plane (R$_\mathrm{E}$)")
    ax.text(0.97, 0.02,
            f"1 Mt detonation ({case['location']}) at {case['altitude']}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=INK, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5))
    return mesh


def main():
    plotter_style()
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 12, "axes.labelweight": "normal",
        "axes.titlesize": 12, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "axes.linewidth": 0.8,
        "axes.spines.top": True, "axes.spines.right": True,
    })
    results = [compute_case(case) for case in CASES]
    vmax = max(result["vmax"] for result in results)
    norm = LogNorm(0.1, vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.5), sharex=True, sharey=True)
    meshes = [draw_panel(ax, result, norm) for ax, result in zip(axes, results)]
    axes[0].set_ylabel(r"Distance along magnetic axis (R$_\mathrm{E}$)")
    axes[0].text(-0.10, 1.03, "a", transform=axes[0].transAxes,
                 fontsize=12, fontweight="bold")
    axes[1].text(-0.10, 1.03, "b", transform=axes[1].transAxes,
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.88, wspace=0.16)
    colorbar_ax = fig.add_axes([0.91, 0.16, 0.015, 0.70])
    colorbar = fig.colorbar(meshes[0], cax=colorbar_ax)
    ticks = [t for t in (0.1, 1, 10, 100) if t < vmax * 0.8] + [vmax]   # top of the scale labelled with its value
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{t:g}" for t in ticks[:-1]] + [f"{vmax:.0f}"])
    colorbar.minorticks_off()
    colorbar.set_label("Peak belt dose rate [rad/s]")
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig2_belt_overlays.{extension}"),
                    dpi=320 if extension == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("Fig_2: wrote fig2_belt_overlays.png/.pdf to", HERE)


if __name__ == "__main__":
    main()
