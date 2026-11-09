"""Debris expansion region built from IGRF field lines.

Steps 3-5 below are our own construction; the other pieces follow the cited sources.

1. Debris energy and mass (Maxwellian mean speed V_D, <v^2> = (3 pi/8) V_D^2):
       E_KE = f_KE * Y,   M_d = 16 E_KE / (3 pi V_D^2).
2. Magnetic confinement radius (Winske 1991 Eq. 1.2, SI):
       R_3 = (3 mu0 E_KE / (2 pi B_b^2))^(1/3),  R_B = 0.8 R_3.
3. Cross-field extent (after LA-UR-12-23076 slide 13; rho_ion is the LANL
   ionized-air profile). In the plane perpendicular to B at the burst, the debris disk is
   a half-ellipse pair: horizontal semi-axis R_h = min(R_B, R_M,h), with
   M_d = (4/3) pi R_M,h^3 rho_ion(h_b); upward semi-axis R_up = min(R_B, R_M,up)
   and downward R_dn = min(R_B, R_M,dn), where R_M,up (R_M,dn) is the radius at
   which the upper (lower) hemisphere about the burst holds M_d/2 of ionized air
       M_d / 2 = integral_0^R integral_hemisphere rho_ion(h(r, theta)) r^2 dOmega dr,
   with spherical-Earth altitude h(r, theta).
   "Up" in that plane is the local vertical projected perpendicular to B.
4. Field-aligned extent (real IGRF lines in place of the ESM spheroid): the debris disk of radius R_h perpendicular to
   B at the burst is divided into N equal-area cells. The IGRF line through
   each cell centre is traced to both feet. Along each direction, the
   ionized-air mass swept per unit disk area is
       m(s) = integral_0^s rho_ion(x(s')) * B_disk / B(s') ds'
   (flux-tube cross-section grows as 1/B by flux conservation). Debris going
   that way stops where m(s) = M_d / (2 pi R_h^2), i.e. each half of the debris
   (up-going and down-going along B) sweeps its own mass, applied cell by
   cell (the equal-mass rule of slide 13 applied per flux tube). If the
   foot (h_lo) comes first, the debris reaches the atmosphere.
5. Debris distribution inside the region (two options):
   "arc" (central): each cell's tube carries equal debris, spread uniformly in
       arc length (field-aligned streaming at constant speed);
   "volume": uniform per unit volume (weight B_disk/B ds), as in the ESM.
6. S on the region (TRH Eq. 7-5): each traced line gives its
   own B_min and foot fields, so for every point on that line
       S = sqrt(1 - B(x)/B_c),  B_c = min(shell B_c(B_min), the line's own feet),
   exactly, without re-tracing. Open lines and points below h_lo have S = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import core as C
from . import physics as P

KT_J = C.KT_J


@dataclass
class Burst:
    lat_deg: float
    lon_deg: float
    alt_km: float
    yield_kt: float = 1000.0
    f_ke: float = 0.25
    v_d_kms: float = 1500.0
    fission_fraction: float = 1.0  # fissions per kt is applied by the caller; eps is per beta


@njit(cache=True)
def _maglat(x, y, z, nx, ny, nz):
    r = math.sqrt(x * x + y * y + z * z)
    s = (x * nx + y * ny + z * nz) / r
    if s > 1.0:
        s = 1.0
    if s < -1.0:
        s = -1.0
    return math.asin(s)


@njit(cache=True)
def _cut_line(path, n, Bdisk, m_target, nx, ny, nz):
    """Arc index/fraction where swept ionized mass per unit disk area reaches m_target.

    Returns (s_stop, reached_atmosphere). Mass in kg/m^2, path s in km.
    """
    m = 0.0
    for k in range(1, n):
        xm = 0.5 * (path[k, 0] + path[k - 1, 0])
        ym = 0.5 * (path[k, 1] + path[k - 1, 1])
        zm = 0.5 * (path[k, 2] + path[k - 1, 2])
        bm = 0.5 * (path[k, 3] + path[k - 1, 3])
        ds = (path[k, 4] - path[k - 1, 4]) * 1e3
        alt = C.geodetic_alt(xm, ym, zm)
        rho = P.rho_ion_lanl(alt, _maglat(xm, ym, zm, nx, ny, nz))
        dm = rho * Bdisk / bm * ds
        if m + dm >= m_target:
            f = (m_target - m) / dm if dm > 0 else 0.0
            return path[k - 1, 4] + f * (path[k, 4] - path[k - 1, 4]), False
        m += dm
    return path[n - 1, 4], True


def hemisphere_mass_radius(M_half, h_b, maglat, up=True, r_max=20000.0):
    """Radius (km) at which the upper/lower hemisphere about the burst holds M_half kg of ionized air."""
    Rb = C.R_E + h_b
    mu = np.linspace(0.0, 1.0, 64) if up else np.linspace(-1.0, 0.0, 64)   # cos(theta) from vertical
    r = np.geomspace(0.1, r_max, 1500)
    hh = np.sqrt(Rb ** 2 + r[:, None] ** 2 + 2 * Rb * r[:, None] * mu[None, :]) - C.R_E
    rho = np.vectorize(P.rho_ion_lanl)(hh, maglat)
    shell = 2 * np.pi * np.trapezoid(rho, mu, axis=1) * (r * 1e3) ** 2     # kg per m of radius
    m = np.concatenate([[0.0], np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(r) * 1e3)])
    if m[-1] < M_half:
        return r_max
    return float(np.interp(M_half, m, r))


def _interp_path(path, n, s):
    """Positions and |B| at arc lengths s (array) on a stored path."""
    sp = path[:n, 4]
    out = np.empty((len(s), 4))
    for c in range(4):
        out[:, c] = np.interp(s, sp, path[:n, c])
    return out


@dataclass
class Region:
    burst: Burst
    E_ke: float
    M_d: float
    B_b: float          # T
    R3: float           # km
    RB: float           # km
    RMh: float          # km
    Rh: float           # km
    Rup: float = 0.0    # km
    Rdn: float = 0.0    # km
    RMup: float = 0.0
    RMdn: float = 0.0
    lines: list = field(default_factory=list)   # per cell dicts
    cell_w: np.ndarray = None                   # debris share per cell
    frac_open: float = 0.0
    frac_reach_atm: float = 0.0


def build_region(burst: Burst, table, n_cells=96, ngrid_s=400, step_frac=0.002, step_cap=10.0,
                 rb_factor=0.8, rh_override=None):
    """Trace the IGRF flux-tube expansion region. `table` is a shells.ShellTable."""
    G, H, nmax = table.G, table.H, table.nmax
    nx, ny, nz = table.nhat
    E_ke = burst.f_ke * burst.yield_kt * KT_J
    V = burst.v_d_kms * 1e3
    M_d = 16.0 * E_ke / (3.0 * math.pi * V * V)
    lat, lon = math.radians(burst.lat_deg), math.radians(burst.lon_deg)
    x0 = np.array(C.geodetic_to_ecef(lat, lon, burst.alt_km))
    bvec = np.array(C.field_ecef(x0[0], x0[1], x0[2], G, H, nmax))
    B_b = np.linalg.norm(bvec) * 1e-9
    R3 = (3.0 * C.MU0 * E_ke / (2.0 * math.pi * B_b ** 2)) ** (1.0 / 3.0) / 1e3
    RB = rb_factor * R3
    rho_b = P.rho_ion_lanl(burst.alt_km, _maglat(x0[0], x0[1], x0[2], nx, ny, nz))
    RMh = (3.0 * M_d / (4.0 * math.pi * rho_b)) ** (1.0 / 3.0) / 1e3
    Rh = rh_override if rh_override is not None else min(RB, RMh)
    mlat = _maglat(x0[0], x0[1], x0[2], nx, ny, nz)
    RMup = hemisphere_mass_radius(0.5 * M_d, burst.alt_km, mlat, up=True)
    RMdn = hemisphere_mass_radius(0.5 * M_d, burst.alt_km, mlat, up=False)
    Rup, Rdn = min(RB, RMup), min(RB, RMdn)
    if rh_override is not None:
        Rup = Rdn = rh_override
    reg = Region(burst, E_ke, M_d, B_b, R3, RB, RMh, Rh, Rup, Rdn, RMup, RMdn)

    # disk basis perpendicular to B (b points along +B)
    b = bvec / np.linalg.norm(bvec)
    up = x0 / np.linalg.norm(x0)                     # local vertical (geocentric)
    eu = up - np.dot(up, b) * b
    if np.linalg.norm(eu) < 1e-6:                   # B vertical: any perpendicular direction
        eu = np.cross(b, [1.0, 0.0, 0.0])
    eu /= np.linalg.norm(eu)
    eh = np.cross(b, eu)
    # equal-area cells on the unit disk (Fibonacci/sunflower, deterministic), then
    # stretched: horizontal by R_h, upper half by R_up, lower half by R_dn.
    k = np.arange(n_cells) + 0.5
    rr = np.sqrt(k / n_cells)
    th = k * math.pi * (3.0 - math.sqrt(5.0))
    cu, ch = rr * np.sin(th), rr * np.cos(th)
    su = np.where(cu >= 0, Rup, Rdn)
    starts = x0[None, :] + (ch * Rh)[:, None] * eh + (cu * su)[:, None] * eu
    cell_area = (math.pi / n_cells) * Rh * su * 1e6           # m^2: each unit-disk cell has area pi/n
    reg.cell_w = cell_area / cell_area.sum()                    # debris share per cell (uniform over disk)
    disk_area = 0.5 * math.pi * Rh * (Rup + Rdn) * 1e6
    m_target = M_d / (2.0 * disk_area)   # kg per m^2 of disk, per direction (each half sweeps M_d/2)
    buf_p = np.zeros((400000, 5)); buf_m = np.zeros((400000, 5))
    n_open = 0; n_atm = 0
    for i, p in enumerate(starts):
        cell = {"start": p, "below": False}
        if C.geodetic_alt(p[0], p[1], p[2]) < table.h_lo:
            cell["below"] = True; reg.lines.append(cell); continue
        Bd = C.field_mag(p[0], p[1], p[2], G, H, nmax)
        res = {}
        for sgn, buf in ((1.0, buf_p), (-1.0, buf_m)):
            n, closed, bhi, blo, bmin, s = C.trace_dir(p[0], p[1], p[2], sgn, G, H, nmax, table.h_lo,
                                                       table.h_hi, step_frac, step_cap, 25 * C.R_E, buf, True)
            s_stop, atm = _cut_line(buf, n, Bd, m_target, nx, ny, nz)
            res[sgn] = (n, closed, bhi, blo, bmin, s_stop, atm)
            # sample positions along the cut piece (fixed count per direction)
            sg = (np.arange(ngrid_s) + 0.5) / ngrid_s * s_stop
            cell[f"pts{int(sgn)}"] = _interp_path(buf, n, sg)
            cell[f"len{int(sgn)}"] = s_stop
            cell[f"atm{int(sgn)}"] = atm
            # whole branch, start to foot: the full flux tube the debris can occupy
            s_foot = buf[n - 1, 4]
            cell[f"full{int(sgn)}"] = _interp_path(buf, n, (np.arange(ngrid_s) + 0.5) / ngrid_s * s_foot)
            cell[f"foot_s{int(sgn)}"] = s_foot
        closed = res[1.0][1] and res[-1.0][1]
        cell["closed"] = closed
        cell["Bmin"] = min(res[1.0][4], res[-1.0][4])
        cell["foot_lo"] = min(res[1.0][3], res[-1.0][3])
        cell["foot_hi"] = min(res[1.0][2], res[-1.0][2])
        cell["Bdisk"] = Bd
        n_open += (not closed)
        n_atm += res[1.0][6] + res[-1.0][6]
        reg.lines.append(cell)
    reg.frac_open = n_open / n_cells
    reg.frac_reach_atm = n_atm / (2.0 * n_cells)
    return reg


def region_S(reg: Region, table, hc=120, weighting="arc"):
    """Region points (x,y,z km), their S, and weights (sum 1). Weighting 'arc' or 'volume'."""
    allp, allS, allw = [], [], []
    for ic, cell in enumerate(reg.lines):
        cw = reg.cell_w[ic]
        if cell["below"]:
            allp.append(cell["start"][None, :]); allS.append(np.zeros(1)); allw.append(np.full(1, cw))
            continue
        bc_shell = float(table.Bc_of(cell["Bmin"], hc)) if cell["closed"] else 0.0
        if not np.isfinite(bc_shell):
            bc_shell = 0.0
        loc = cell["foot_hi"] if hc == int(round(table.h_hi)) else cell["foot_lo"]
        bc = min(bc_shell, loc) if bc_shell > 0 else 0.0
        for sgn in (1, -1):
            pts = cell[f"pts{sgn}"]
            Bp = pts[:, 3]
            alt = np.array([C.geodetic_alt(*q[:3]) for q in pts])
            S = np.where((bc > 0) & (Bp < bc) & (alt >= hc),
                         np.sqrt(np.clip(1.0 - Bp / max(bc, 1e-30), 0, 1)), 0.0)
            # half of the cell's debris goes each way along B
            if weighting == "arc":
                w = np.full(len(pts), 0.5 * cw / len(pts))
            else:
                v = cell["Bdisk"] / Bp
                w = 0.5 * cw * v / v.sum()
            allp.append(pts[:, :3]); allS.append(S); allw.append(w)
    return np.vstack(allp), np.concatenate(allS), np.concatenate(allw)
