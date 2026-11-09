"""Shared setup for the paper calculations: paths, trapping source, belt evolution and dose.

Source:  hanetrap expansion region, IGRF-14 at 2026-01-14 (the TLE epoch), loss altitude 120 km.
         For each burst, the 96 cells of the IGRF expansion region give their drift shell
         (L = (B0/B_min)^(1/3), B0 = 31,200 nT) and their volume-averaged trapping S_c over
         the full shell (foot to foot, all longitudes). The trapped fraction per beta is
         binned in L:  Lw(L) = sum_c w_c S_c delta(L - L_c),  sum = <S>.
Betas:   U-235 fission-product beta table; 0.404 betas per fission (all debris down by 2 s),
         1.289 (by 20 s), 5.207 (aloft 6 months, emitted over time following the table).
Belt:    Carter spectrum, Brautigam-Albert D_LL (Kp = 2), empirical lifetimes, dipole
         drift-shell volumes above 80 km, dose behind 2 mm Al (openhane_src).
Fleet:   prompt X-ray model (openhane_src) and CelesTrak TLEs from 2026-01-14.

HANE_YIELD_KT overrides the 1 Mt yield.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

FIGS = os.path.dirname(os.path.abspath(__file__))            # <repo>/analysis
BUNDLE = os.path.dirname(FIGS)                                 # <repo>
HANE = os.path.join(BUNDLE, "hanetrap")
OHSRC = os.path.join(BUNDLE, "openhane_src")
DATA = os.path.join(BUNDLE, "data")
NATURAL_EARTH = os.path.join(DATA, "natural_earth")
TLE_FILE = os.path.join(DATA, "celestrack-20260114-0200CT")
sys.path.insert(0, HANE); sys.path.insert(0, OHSRC)
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
try:                                   # Natural Earth shapefiles shipped in data/ (no download needed)
    import cartopy
    cartopy.config["pre_existing_data_dir"] = NATURAL_EARTH
except Exception:
    pass

from hanetrap import shells, core as C  # noqa: E402
from hanetrap.region import Burst, build_region  # noqa: E402
import tier1_source_belt as T1  # noqa: E402
import tier3_diffusion as T3D  # noqa: E402
import tier3_dose_response as T3R  # noqa: E402

EPOCH = 2026.0 + 13.0 / 365.0           # 2026-01-14
B0_OH = 31200.0                         # nT, dipole reference field used for L
YIELD_KT, FISSION, KP, SHIELD, H_ATM, HC = 1000.0, 0.50, 2.0, 2.0, 80.0, 120
YIELD_KT = float(os.environ.get("HANE_YIELD_KT", YIELD_KT))   # optional override (kt); default 1 Mt
BETAS = {"low": 0.404, "mid": 1.289, "high": 5.207}
FISSIONS = YIELD_KT * FISSION * T1.FISSIONS_PER_KT

# ---- belt grids (finer L resolution for the low shells)
LG = np.round(np.arange(1.02, 6.0 + 1e-9, 0.01), 4)
NL = len(LG); dLg = LG[1] - LG[0]
EG = np.geomspace(0.05, 8.0, 24)
EW = T1.carter_beta_energy_weights(EG)
VEL = T1.electron_speed_m_s(EG)
DLL = T3D.brautigam_albert_DLL_per_s(LG, KP)
TAU = T3D.loss_tau_s_by_name("empirical", EG, LG)
_edges = np.r_[LG[0] - dLg / 2, 0.5 * (LG[1:] + LG[:-1]), LG[-1] + dLg / 2]
VOL = np.array([T3D.drift_shell_volume_bin_m3(max(1.013, _edges[i]), max(1.014, _edges[i + 1]), H_ATM)
                for i in range(NL)])
DRESP = T3R.electron_dose_response_rad_per_e_cm2(EG, SHIELD, True)

_T = None


def table():
    global _T
    if _T is None:
        _T = shells.ShellTable(EPOCH, verbose=False)
    return _T


# ---------------------------------------------------------------- expansion-region cells
def region_cells(lat, lon, alt, yield_kt=YIELD_KT, n_cells=96, fast=False):
    """Per-cell drift-shell L, debris share w and full-shell volume-averaged S.

    fast=True traces the region's field lines with step 0.005 r (cap 25 km) instead of 0.002 r
    (cap 10 km); tested to give identical <S> and peak dose rate to 4 digits, 2.5x faster.
    """
    T = table()
    kw = dict(step_frac=0.005, step_cap=25.0) if fast else {}
    reg = build_region(Burst(lat, lon, alt, yield_kt=yield_kt), T, n_cells=n_cells, **kw)
    nlon = len(T.lons)
    Lc, wc, Sc = [], [], []
    for ic, cell in enumerate(reg.lines):
        w = float(reg.cell_w[ic])
        if cell["below"] or not cell["closed"]:
            Lc.append(np.nan); wc.append(w); Sc.append(0.0); continue
        bmin = cell["Bmin"]
        bc = float(T.Bc_of(bmin, HC))
        if not (np.isfinite(bc) and bc > 0):
            Lc.append((B0_OH / bmin) ** (1 / 3)); wc.append(w); Sc.append(0.0); continue
        num = den = 0.0
        for j in range(nlon):
            m = T.meta[j]; idx = np.nonzero(m[:, 0] > 0.5)[0]
            bl = m[idx, 1]
            if not (bl.min() <= bmin <= bl.max()):
                continue
            k = min(max(np.searchsorted(-bl, -bmin), 1), len(idx) - 1)
            i0, i1 = idx[k - 1], idx[k]
            u = (bl[k - 1] - bmin) / (bl[k - 1] - bl[k]) if bl[k - 1] != bl[k] else 0.0
            B = (1 - u) * T.poly[j, i0][:, 3] + u * T.poly[j, i1][:, 3]
            length = (1 - u) * m[i0, 4] + u * m[i1, 4]
            v = (length / len(B)) / B
            num += float(np.sum(v * np.sqrt(np.clip(1 - B / bc, 0, 1)))); den += float(np.sum(v))
        Lc.append((B0_OH / bmin) ** (1 / 3)); wc.append(w); Sc.append(num / den if den > 0 else 0.0)
    Lc, wc, Sc = map(np.array, (Lc, wc, Sc))
    return dict(L=Lc, w=wc, S=Sc, meanS=float(np.sum(wc * Sc)), reg=reg,
                L_burst=float(np.nanmean(Lc)) if np.isfinite(Lc).any() else np.nan)


def L_weights(cells, smooth=0.0):
    """Trapped fraction per beta binned on LG (linear sharing between neighbours); sums to <S>.

    smooth: optional Gaussian width in L (display/convergence aid). Tests against 1,536 cells show
    192 cells with smooth = 0.01 reproduce the converged peak dose rate within about 10%.
    """
    out = np.zeros(NL)
    for L, w, S in zip(cells["L"], cells["w"], cells["S"]):
        if not np.isfinite(L) or w * S <= 0 or L < LG[0] or L > LG[-1]:
            continue
        j = int(np.searchsorted(LG, L))
        if j == 0:
            out[0] += w * S; continue
        u = (L - LG[j - 1]) / dLg
        out[j - 1] += w * S * (1 - u); out[j] += w * S * u
    if smooth > 0:
        k = np.exp(-0.5 * ((np.arange(-50, 51) * dLg) / smooth) ** 2); k /= k.sum()
        tot = out.sum(); out = np.convolve(out, k, mode="same")
        out *= tot / out.sum() if out.sum() > 0 else 1.0
    return out


# ---------------------------------------------------------------- beta source
def _activity():
    d = os.path.join(HANE, "data") + os.sep
    s = list(csv.DictReader(open(d + "u235_beta_activity_subsecond_per_fission.csv")))
    l = list(csv.DictReader(open(d + "u235_1kt_beta_activity_timeseries_with_seconds.csv")))
    t = [float(r["time_s"]) for r in s]; a = [float(r["beta_activity_s-1_per_fission"]) for r in s]
    for r in l:
        if float(r["time_s"]) >= 1:
            t.append(float(r["time_s"])); a.append(float(r["beta_activity_s-1_per_fission"]))
    t, a = np.array(t), np.array(a)
    k, _ = np.polyfit(np.log(t[t >= 3600]), np.log(a[t >= 3600]), 1)
    return t, a, k


def cumulative_betas(times_s):
    """N(t) per fission from the U-235 beta table (linear < 1 s, log-log above, power tail), vectorized."""
    t, a, k = _activity()
    g = np.r_[np.linspace(0, 1, 2001), np.geomspace(1, max(times_s.max(), 2.0), 60000)[1:]]
    lo = g <= 1
    rate = np.empty_like(g)
    rate[lo] = np.interp(g[lo], t[t <= 1], a[t <= 1])
    mid = (g > 1) & (g <= t[-1])
    rate[mid] = np.exp(np.interp(np.log(g[mid]), np.log(t[t > 0]), np.log(a[t > 0])))
    hi = g > t[-1]
    rate[hi] = a[-1] * (g[hi] / t[-1]) ** k
    cum = np.r_[0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(g))]
    return np.interp(times_s, g, cum)


# ---------------------------------------------------------------- belt
def dose_rate_initial(Lw, beta):
    """Dose rate (rad/s) on LG right after an instantaneous injection of beta betas per fission."""
    N = FISSIONS * beta * EW[:, None] * Lw[None, :]
    flux = N / VOL[None, :] * VEL[:, None] / 1e4
    return np.sum(flux * DRESP[:, None], axis=0)


def evolve(Lw, mode, times_s):
    """Dose rate (rad/s) on (times, LG). mode 'low'/'mid': instantaneous injection at t = 0;
    'high': 0.404 at t = 0, then the rest of the table's emission as a source each step."""
    beta0 = BETAS[mode] if mode != "high" else BETAS["low"]
    N = FISSIONS * beta0 * EW[:, None] * Lw[None, :]
    if mode == "high":
        cum = cumulative_betas(times_s)
        cum = np.minimum(cum, BETAS["high"])               # betas stop at six months
        cum[0] = BETAS["low"]
    out = np.zeros((len(times_s), NL))
    inv_tau = 1.0 / TAU
    for it in range(len(times_s)):
        flux = N / VOL[None, :] * VEL[:, None] / 1e4
        out[it] = np.sum(flux * DRESP[:, None], axis=0)
        if it == len(times_s) - 1:
            break
        dt = times_s[it + 1] - times_s[it]
        src = 0.0
        if mode == "high":
            dbeta = max(cum[it + 1] - cum[it], 0.0)
            src = FISSIONS * dbeta * EW[:, None] * Lw[None, :]   # electrons added this step
        Nn = np.empty_like(N)
        for ie in range(len(EG)):
            Nn[ie] = T3D.implicit_diffusion_step(N[ie], LG, DLL, dt, inv_tau=inv_tau[ie],
                                                 source=(src[ie] if mode == "high" else 0.0))
        N = np.maximum(Nn, 0.0)
    return out


# ---------------------------------------------------------------- where on the shell the electrons are
def shell_Sbar(Ls):
    """Volume-averaged S over the full drift shell (all longitudes, foot to foot) for each L.

    Trapped electrons that are isotropic outside the loss cone have local density proportional to the
    local trapped solid-angle fraction S(x) = sqrt(1 - B(x)/B_c) (Liouville). The belt model stores the
    shell-average density, so the density at a point is  n(x) = n_avg(L) * S(x) / Sbar(L).
    """
    T = table()
    out = np.zeros(len(Ls))
    for q, L in enumerate(Ls):
        bmin = B0_OH / L ** 3
        bc = float(T.Bc_of(bmin, HC))
        if not (np.isfinite(bc) and bc > 0):
            continue
        num = den = 0.0
        for j in range(len(T.lons)):
            m = T.meta[j]; idx = np.nonzero(m[:, 0] > 0.5)[0]; bl = m[idx, 1]
            if not (bl.min() <= bmin <= bl.max()):
                continue
            k = min(max(np.searchsorted(-bl, -bmin), 1), len(idx) - 1)
            i0, i1 = idx[k - 1], idx[k]
            u = (bl[k - 1] - bmin) / (bl[k - 1] - bl[k]) if bl[k - 1] != bl[k] else 0.0
            B = (1 - u) * T.poly[j, i0][:, 3] + u * T.poly[j, i1][:, 3]
            length = (1 - u) * m[i0, 4] + u * m[i1, 4]
            v = (length / len(B)) / B
            num += float(np.sum(v * np.sqrt(np.clip(1 - B / bc, 0, 1)))); den += float(np.sum(v))
        out[q] = num / den if den > 0 else 0.0
    return out


def local_S(P_ecef, Ls):
    """S = sqrt(1 - B/B_c) at ECEF points P (km), with B_c of the shell L (B_min = 31,200/L^3 nT)."""
    T = table()
    B = np.linalg.norm(C.batch_field(P_ecef, T.G, T.H, T.nmax), axis=1)
    bc = np.nan_to_num(T.Bc_of(B0_OH / np.asarray(Ls) ** 3, HC), nan=0.0)
    return np.where(bc > 0, np.sqrt(np.clip(1 - B / np.where(bc > 0, bc, 1.0), 0, 1)), 0.0)
