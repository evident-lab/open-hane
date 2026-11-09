"""Core numerics: IGRF field, WGS-84 geodesy, field-line tracing.

Every routine here is compiled with numba. Units: km for lengths, nT for B,
radians inside functions.
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import njit

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ---------------------------------------------------------------- constants
# CODATA 2018 (NIST)
E_CHARGE = 1.602176634e-19      # C
EPS0 = 8.8541878128e-12         # F/m
MU0 = 1.25663706212e-6          # N/A^2
A0 = 5.29177210903e-11          # m, Bohr radius
HBAR = 1.054571817e-34          # J s
AMU = 1.66053906660e-27         # kg
KE = 1.0 / (4.0 * math.pi * EPS0)
V0 = KE * E_CHARGE ** 2 / HBAR  # Bohr velocity, 2.188e6 m/s
KT_J = 4.184e12                 # J per kiloton

A_IGRF = 6371.2                 # km, IGRF reference radius (Alken et al. 2021)
WGS_A = 6378.137                # km, WGS-84 (NIMA TR8350.2)
WGS_F = 1.0 / 298.257223563
WGS_E2 = WGS_F * (2.0 - WGS_F)
WGS_B = WGS_A * (1.0 - WGS_F)
R_E = 6371.2                    # km, used only for "Earth radii" guards


# ---------------------------------------------------------------- IGRF coefficients
def load_igrf(epoch_year: float, fn: str | None = None):
    """Gauss coefficients g[n,m], h[n,m] (nT) at a decimal year.

    Reads the official IGRF-14 SHC file and interpolates linearly between the
    5-year models, the interpolation rule the IGRF itself specifies.
    """
    fn = fn or os.path.join(DATA, "IGRF14.shc")
    rows = []
    years = None
    with open(fn) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if years is None:
                if len(parts) == 7 and parts[0] == "1":   # header line
                    continue
                years = np.array([float(x) for x in parts])
                continue
            rows.append([float(x) for x in parts])
    rows = np.array(rows)
    if not years[0] <= epoch_year <= years[-1]:
        raise ValueError("epoch outside IGRF range")
    i = min(np.searchsorted(years, epoch_year, side="right") - 1, len(years) - 2)
    w = (epoch_year - years[i]) / (years[i + 1] - years[i])
    vals = rows[:, 2 + i] * (1 - w) + rows[:, 3 + i] * w
    nmax = int(rows[:, 0].max())
    G = np.zeros((nmax + 1, nmax + 1))
    H = np.zeros((nmax + 1, nmax + 1))
    for (n, m), v in zip(rows[:, :2].astype(int), vals):
        if m >= 0:
            G[n, m] = v
        else:
            H[n, -m] = v
    # trim trailing all-zero degrees (pre-2000 models stop at n = 10)
    while nmax > 1 and not np.any(G[nmax]) and not np.any(H[nmax]):
        nmax -= 1
    return G[: nmax + 1, : nmax + 1].copy(), H[: nmax + 1, : nmax + 1].copy(), nmax


def dipole_axis(G, H):
    """Unit vector toward the north geomagnetic pole and B0 = |dipole| (nT)."""
    g10, g11, h11 = G[1, 0], G[1, 1], H[1, 1]
    b0 = math.sqrt(g10 ** 2 + g11 ** 2 + h11 ** 2)
    n = -np.array([g11, h11, g10]) / b0
    return n, b0


# ---------------------------------------------------------------- IGRF field
@njit(cache=True)
def field_ecef(x, y, z, G, H, nmax):
    """IGRF field (nT) in ECEF Cartesian components at an ECEF point (km)."""
    r = math.sqrt(x * x + y * y + z * z)
    c = z / r
    rho = math.sqrt(x * x + y * y)
    s = rho / r
    if s < 1e-12:
        s = 1e-12
    phi = math.atan2(y, x)
    P = np.zeros((nmax + 1, nmax + 1))
    dP = np.zeros((nmax + 1, nmax + 1))
    P[0, 0] = 1.0
    for n in range(1, nmax + 1):
        for m in range(0, n + 1):
            if n == m:
                if n == 1:
                    P[1, 1] = s
                    dP[1, 1] = c
                else:
                    k = math.sqrt((2.0 * n - 1.0) / (2.0 * n))
                    P[n, n] = k * s * P[n - 1, n - 1]
                    dP[n, n] = k * (s * dP[n - 1, n - 1] + c * P[n - 1, n - 1])
            else:
                a1 = (2.0 * n - 1.0) / math.sqrt(n * n - m * m)
                p2 = 0.0
                d2 = 0.0
                b1 = 0.0
                if n - 2 >= m:
                    b1 = math.sqrt(((n - 1.0) ** 2 - m * m) / (n * n - m * m))
                    p2 = P[n - 2, m]
                    d2 = dP[n - 2, m]
                P[n, m] = a1 * c * P[n - 1, m] - b1 * p2
                dP[n, m] = a1 * (c * dP[n - 1, m] - s * P[n - 1, m]) - b1 * d2
    br = 0.0
    bt = 0.0
    bp = 0.0
    ar = A_IGRF / r
    arn = ar * ar  # (a/r)^(n+2) starts at n=0 -> (a/r)^2
    for n in range(1, nmax + 1):
        arn *= ar
        for m in range(0, n + 1):
            cm = math.cos(m * phi)
            sm = math.sin(m * phi)
            t = G[n, m] * cm + H[n, m] * sm
            br += (n + 1.0) * arn * t * P[n, m]
            bt -= arn * t * dP[n, m]
            bp += arn * m * (G[n, m] * sm - H[n, m] * cm) * P[n, m] / s
    cp = math.cos(phi)
    sp = math.sin(phi)
    bx = (br * s + bt * c) * cp - bp * sp
    by = (br * s + bt * c) * sp + bp * cp
    bz = br * c - bt * s
    return bx, by, bz


@njit(cache=True)
def field_mag(x, y, z, G, H, nmax):
    bx, by, bz = field_ecef(x, y, z, G, H, nmax)
    return math.sqrt(bx * bx + by * by + bz * bz)


# ---------------------------------------------------------------- geodesy
@njit(cache=True)
def geodetic_to_ecef(lat, lon, h):
    """Geodetic lat, lon (radians) and height h (km) -> ECEF km, WGS-84."""
    sl = math.sin(lat)
    N = WGS_A / math.sqrt(1.0 - WGS_E2 * sl * sl)
    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N * (1.0 - WGS_E2) + h) * sl
    return x, y, z


@njit(cache=True)
def geodetic_alt(x, y, z):
    """Geodetic altitude (km) by Bowring's method, two iterations (mm-level)."""
    p = math.sqrt(x * x + y * y)
    ep2 = (WGS_A ** 2 - WGS_B ** 2) / WGS_B ** 2
    th = math.atan2(z * WGS_A, p * WGS_B)
    lat = math.atan2(z + ep2 * WGS_B * math.sin(th) ** 3,
                     p - WGS_E2 * WGS_A * math.cos(th) ** 3)
    for _ in range(2):
        N = WGS_A / math.sqrt(1.0 - WGS_E2 * math.sin(lat) ** 2)
        h = p * math.cos(lat) + z * math.sin(lat) - WGS_A * math.sqrt(1.0 - WGS_E2 * math.sin(lat) ** 2)
        lat = math.atan2(z, p * (1.0 - WGS_E2 * N / (N + h)))
    return p * math.cos(lat) + z * math.sin(lat) - WGS_A * math.sqrt(1.0 - WGS_E2 * math.sin(lat) ** 2)


@njit(cache=True)
def geodetic_lat(x, y, z):
    p = math.sqrt(x * x + y * y)
    ep2 = (WGS_A ** 2 - WGS_B ** 2) / WGS_B ** 2
    th = math.atan2(z * WGS_A, p * WGS_B)
    return math.atan2(z + ep2 * WGS_B * math.sin(th) ** 3,
                      p - WGS_E2 * WGS_A * math.cos(th) ** 3)


# ---------------------------------------------------------------- tracing
@njit(cache=True)
def _dirn(x, y, z, sign, G, H, nmax):
    bx, by, bz = field_ecef(x, y, z, G, H, nmax)
    b = math.sqrt(bx * bx + by * by + bz * bz)
    return sign * bx / b, sign * by / b, sign * bz / b


@njit(cache=True)
def trace_dir(x0, y0, z0, sign, G, H, nmax, h_lo, h_hi, step_frac, step_cap,
              max_r, path, store):
    """Trace one direction of a field line with fixed-fraction RK4 steps.

    Stops where geodetic altitude falls below h_lo. Records |B| where the line
    first crosses h_hi and h_lo, the minimum |B| on the way, and (if store)
    every node in path[k] = (x, y, z, |B|, s).

    Returns (n_nodes, closed, B_hi, B_lo, Bmin, s_total).
    """
    x, y, z = x0, y0, z0
    alt = geodetic_alt(x, y, z)
    bcur = field_mag(x, y, z, G, H, nmax)
    bmin = bcur
    b_hi = -1.0
    b_lo = -1.0
    if alt < h_hi:          # start already below h_hi: the h_hi crossing is at the start
        b_hi = bcur
    s = 0.0
    n = 0
    if store:
        path[0, 0] = x; path[0, 1] = y; path[0, 2] = z; path[0, 3] = bcur; path[0, 4] = 0.0
    n = 1
    maxn = path.shape[0]
    steps = 0
    while True:
        steps += 1
        r = math.sqrt(x * x + y * y + z * z)
        h = step_frac * r
        if h > step_cap:
            h = step_cap
        # refine steps close to the atmosphere so the crossing is resolved
        if alt < h_hi + 50.0 and h > 5.0:
            h = 5.0
        k1x, k1y, k1z = _dirn(x, y, z, sign, G, H, nmax)
        k2x, k2y, k2z = _dirn(x + 0.5 * h * k1x, y + 0.5 * h * k1y, z + 0.5 * h * k1z, sign, G, H, nmax)
        k3x, k3y, k3z = _dirn(x + 0.5 * h * k2x, y + 0.5 * h * k2y, z + 0.5 * h * k2z, sign, G, H, nmax)
        k4x, k4y, k4z = _dirn(x + h * k3x, y + h * k3y, z + h * k3z, sign, G, H, nmax)
        nx = x + h * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        ny = y + h * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        nz = z + h * (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
        nalt = geodetic_alt(nx, ny, nz)
        nb = field_mag(nx, ny, nz, G, H, nmax)
        if b_hi < 0.0 and nalt < h_hi and alt >= h_hi:
            u = (alt - h_hi) / (alt - nalt)
            b_hi = field_mag(x + u * (nx - x), y + u * (ny - y), z + u * (nz - z), G, H, nmax)
        if nalt < h_lo:
            u = (alt - h_lo) / (alt - nalt)
            fx = x + u * (nx - x)
            fy = y + u * (ny - y)
            fz = z + u * (nz - z)
            b_lo = field_mag(fx, fy, fz, G, H, nmax)
            if b_hi < 0.0:
                b_hi = b_lo
            s += u * h
            if store and n < maxn:
                path[n, 0] = fx; path[n, 1] = fy; path[n, 2] = fz; path[n, 3] = b_lo; path[n, 4] = s
                n += 1
            return n, True, b_hi, b_lo, bmin, s
        x, y, z = nx, ny, nz
        alt = nalt
        s += h
        if nb < bmin:
            bmin = nb
        if store and n < maxn:
            path[n, 0] = x; path[n, 1] = y; path[n, 2] = z; path[n, 3] = nb; path[n, 4] = s
            n += 1
        if math.sqrt(x * x + y * y + z * z) > max_r or steps > 200000 or (store and n >= maxn):
            return n, False, b_hi, b_lo, bmin, s


@njit(cache=True)
def line_summary(x0, y0, z0, G, H, nmax, h_lo, h_hi, step_frac, step_cap, max_r, buf):
    """Trace both directions; return (closed, Bmin, Bfoot_lo_min, Bfoot_hi_min, B0).

    Bfoot_*_min is the smaller of the two footpoint fields at that altitude:
    the local bounce loss cone (LANL "Bounce" criterion, C&C 2017 p. 47).
    """
    b0 = field_mag(x0, y0, z0, G, H, nmax)
    if geodetic_alt(x0, y0, z0) < h_lo:
        return False, b0, 0.0, 0.0, b0
    n1, c1, hi1, lo1, m1, s1 = trace_dir(x0, y0, z0, 1.0, G, H, nmax, h_lo, h_hi,
                                         step_frac, step_cap, max_r, buf, False)
    n2, c2, hi2, lo2, m2, s2 = trace_dir(x0, y0, z0, -1.0, G, H, nmax, h_lo, h_hi,
                                         step_frac, step_cap, max_r, buf, False)
    closed = c1 and c2
    bmin = min(m1, m2)
    return closed, bmin, min(lo1, lo2), min(hi1, hi2), b0


def batch_field(points, G, H, nmax):
    out = np.empty((len(points), 3))
    _batch_field(np.ascontiguousarray(points, dtype=np.float64), G, H, nmax, out)
    return out


@njit(cache=True)
def _batch_field(P, G, H, nmax, out):
    for i in range(P.shape[0]):
        out[i, 0], out[i, 1], out[i, 2] = field_ecef(P[i, 0], P[i, 1], P[i, 2], G, H, nmax)
