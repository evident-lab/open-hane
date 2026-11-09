"""Drift-shell loss table and the trapping function S(x).

Approximation (TRH p. 3-33): drift shells are labelled by the
minimum field B_eq on the field line. All lines with the same B_eq, one per
longitude, form the shell. TRH states drift-shell splitting is under 1% of the
shell radius below L ~ 3, so this is the TRH Eq. 7-5 treatment, not the full
(B_m, K) construction of Cunningham & Cowee (2017). Results above L ~ 3 carry
that caveat.

S(x) = sqrt(1 - B(x)/B_c)  (TRH Eq. 7-5, printed p. 7-9),
B_c = smallest field at the loss altitude anywhere on the drift shell.
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import njit, prange

from . import core as C

NDEC = 64  # points kept per field line for shell sampling


@njit(parallel=True, cache=True)
def _build_lines(starts, G, H, nmax, h_lo, h_hi, step_frac, step_cap, max_r,
                 out_meta, out_poly):
    """Trace each start both ways. meta = (closed, Bmin, Bfoot_lo, Bfoot_hi, length)."""
    M = starts.shape[0]
    for i in prange(M):
        buf1 = np.zeros((60000, 5))
        buf2 = np.zeros((60000, 5))
        x0, y0, z0 = starts[i, 0], starts[i, 1], starts[i, 2]
        n1, c1, hi1, lo1, m1, s1 = C.trace_dir(x0, y0, z0, 1.0, G, H, nmax, h_lo, h_hi,
                                               step_frac, step_cap, max_r, buf1, True)
        n2, c2, hi2, lo2, m2, s2 = C.trace_dir(x0, y0, z0, -1.0, G, H, nmax, h_lo, h_hi,
                                               step_frac, step_cap, max_r, buf2, True)
        closed = c1 and c2
        out_meta[i, 0] = 1.0 if closed else 0.0
        out_meta[i, 1] = min(m1, m2)
        out_meta[i, 2] = min(lo1, lo2)
        out_meta[i, 3] = min(hi1, hi2)
        out_meta[i, 4] = s1 + s2
        if not closed:
            continue
        # combined path from the -B foot to the +B foot, arc length from -B foot
        tot = s1 + s2
        for k in range(NDEC):
            target = tot * (k + 0.5) / NDEC
            if target <= s2:          # on the -B branch, measured back from its foot
                sb = s2 - target
                buf = buf2; n = n2
            else:
                sb = target - s2
                buf = buf1; n = n1
            j = 1
            while j < n - 1 and buf[j, 4] < sb:
                j += 1
            ds = buf[j, 4] - buf[j - 1, 4]
            u = 0.0 if ds <= 0 else (sb - buf[j - 1, 4]) / ds
            for c in range(4):
                out_poly[i, k, c] = buf[j - 1, c] + u * (buf[j, c] - buf[j - 1, c])


class ShellTable:
    """B_c(B_eq) for loss altitudes h_lo and h_hi (default 100 and 120 km), plus stored shell lines."""

    def __init__(self, epoch_year, n_lon=72, step_frac=0.004, step_cap=25.0,
                 h_lo=100.0, h_hi=120.0, cache_dir=None, verbose=True):
        self.G, self.H, self.nmax = C.load_igrf(epoch_year)
        self.epoch = epoch_year
        self.h_lo, self.h_hi = h_lo, h_hi
        self.nhat, self.B0 = C.dipole_axis(self.G, self.H)
        key = f"shells_{epoch_year:.4f}_{n_lon}_{step_frac}_{step_cap}_{h_lo}_{h_hi}_ndec{NDEC}.npz"
        cache_dir = cache_dir or os.path.join(os.path.dirname(C.DATA), "outputs", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        fn = os.path.join(cache_dir, key)
        if os.path.exists(fn):
            d = np.load(fn)
            self.lons, self.Lk, meta, poly = d["lons"], d["Lk"], d["meta"], d["poly"]
        else:
            self.lons = np.linspace(-math.pi, math.pi, n_lon, endpoint=False)
            # L-like grid for start radii: dense at low L where the loss cone changes fast
            self.Lk = np.unique(np.r_[np.arange(1.010, 1.5, 0.002), np.arange(1.5, 3.0, 0.01),
                                      np.arange(3.0, 8.01, 0.05)])
            starts = []
            for lon in self.lons:
                nx, ny, nz = self.nhat
                lat = math.atan2(-(nx * math.cos(lon) + ny * math.sin(lon)), nz)  # dipole equator
                for L in self.Lk:
                    r = L * C.R_E
                    starts.append([r * math.cos(lat) * math.cos(lon), r * math.cos(lat) * math.sin(lon),
                                   r * math.sin(lat)])
            starts = np.array(starts)
            meta = np.zeros((len(starts), 5))
            poly = np.zeros((len(starts), NDEC, 4))
            if verbose:
                print(f"tracing {len(starts)} shell lines (epoch {epoch_year:.3f})", flush=True)
            _build_lines(starts, self.G, self.H, self.nmax, h_lo, h_hi, step_frac, step_cap,
                         25 * C.R_E, meta, poly)
            np.savez_compressed(fn, lons=self.lons, Lk=self.Lk, meta=meta, poly=poly)
        nL = len(self.Lk)
        self.meta = meta.reshape(len(self.lons), nL, 5)
        self.poly = poly.reshape(len(self.lons), nL, NDEC, 4)
        self._build_curves()

    def _build_curves(self):
        # B_eq grid, log spaced over the range covered at every longitude
        self.Bgrid = np.geomspace(40.0, 40000.0, 3000)
        nl = len(self.lons)
        bc = {0: np.full((nl, len(self.Bgrid)), np.nan), 1: np.full((nl, len(self.Bgrid)), np.nan)}
        # start altitudes (km) of each stored line, to drop lines that begin inside the atmosphere
        nx, ny, nz = self.nhat
        start_alt = np.empty((nl, len(self.Lk)))
        for j, lon in enumerate(self.lons):
            lat = math.atan2(-(nx * math.cos(lon) + ny * math.sin(lon)), nz)
            for k, L in enumerate(self.Lk):
                r = L * C.R_E
                start_alt[j, k] = C.geodetic_alt(r * math.cos(lat) * math.cos(lon),
                                                 r * math.cos(lat) * math.sin(lon), r * math.sin(lat))
        bad = (start_alt < self.h_hi) | (self.meta[:, :, 2] < self.meta[:, :, 1])
        self.meta[:, :, 0] = np.where(bad, 0.0, self.meta[:, :, 0])
        for j in range(nl):
            m = self.meta[j]
            ok = m[:, 0] > 0.5
            bmin = m[ok, 1]
            order = np.argsort(bmin)
            bmin = bmin[order]
            for col, key in ((2, 0), (3, 1)):
                bf = m[ok, col][order]
                v = np.interp(self.Bgrid, bmin, bf, left=np.nan, right=0.0)
                # B_eq stronger than any closed line at this longitude: the shell
                # dips into the atmosphere here, so nothing on it is trapped
                bc[key][j] = v
        self.Bc_lon = bc
        lo, hi = int(round(self.h_lo)), int(round(self.h_hi))   # loss altitudes this table supports
        self.Bc = {lo: np.min(bc[0], axis=0), hi: np.min(bc[1], axis=0)}
        self.argmin_lon = {lo: np.argmin(np.nan_to_num(bc[0], nan=1e12), axis=0),
                           hi: np.argmin(np.nan_to_num(bc[1], nan=1e12), axis=0)}

    def Bc_of(self, Beq, hc=100):
        return np.interp(Beq, self.Bgrid, self.Bc[hc], left=np.nan, right=0.0)

    def L_of(self, Beq):
        return (self.B0 / np.asarray(Beq)) ** (1.0 / 3.0)

    def dlc_angle_deg(self, L, hc=120):
        """Equatorial drift-loss-cone angle: sin^2(a) = B_eq / B_c."""
        Beq = self.B0 / np.asarray(L, float) ** 3
        Bc = self.Bc_of(Beq, hc)
        ratio = np.clip(Beq / np.where(Bc > 0, Bc, np.nan), 0, 1)
        return np.degrees(np.arcsin(np.sqrt(ratio)))


@njit(parallel=True, cache=True)
def _S_batch(P, G, H, nmax, h_lo, h_hi, use_hi, Bgrid, Bc, step_frac, step_cap, max_r,
             out_S, out_Bmin, out_Bloc):
    for i in prange(P.shape[0]):
        buf = np.zeros((1, 5))
        closed, bmin, blo, bhi, b0 = C.line_summary(P[i, 0], P[i, 1], P[i, 2], G, H, nmax,
                                                    h_lo, h_hi, step_frac, step_cap, max_r, buf)
        out_Bmin[i] = bmin
        out_Bloc[i] = blo if use_hi == 0 else bhi
        if not closed:
            out_S[i] = 0.0
            continue
        # shell value (global min over longitudes) by interpolation in B_eq
        if bmin < Bgrid[0]:
            out_S[i] = 0.0          # beyond table (L > ~8): outside model validity
            continue
        if bmin > Bgrid[-1]:
            out_S[i] = 0.0
            continue
        j = np.searchsorted(Bgrid, bmin)
        if j < 1:
            j = 1
        u = (bmin - Bgrid[j - 1]) / (Bgrid[j] - Bgrid[j - 1])
        bc = Bc[j - 1] + u * (Bc[j] - Bc[j - 1])
        if not (bc > 0.0):
            out_S[i] = 0.0
            continue
        loc = blo if use_hi == 0 else bhi
        if loc < bc:
            bc = loc                # the local line's own feet can be lower than sampled shell min
        if b0 >= bc:
            out_S[i] = 0.0
        else:
            out_S[i] = math.sqrt(1.0 - b0 / bc)


def S_points(table: ShellTable, points, hc=100, step_frac=0.004, step_cap=25.0):
    """Trapping function S at ECEF points (km). Returns S, B_eq of the line, B_local_feet."""
    P = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    S = np.zeros(len(P)); Bmin = np.zeros(len(P)); Bloc = np.zeros(len(P))
    bc = np.nan_to_num(table.Bc[hc], nan=-1.0)
    _S_batch(P, table.G, table.H, table.nmax, table.h_lo, table.h_hi, 1 if hc == int(round(table.h_hi)) else 0,
             table.Bgrid, bc, step_frac, step_cap, 25 * C.R_E, S, Bmin, Bloc)
    return S, Bmin, Bloc
