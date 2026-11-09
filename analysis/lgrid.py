"""Satellite L lookup grid, L = (31,200 nT / B_min)^(1/3) at IGRF-14 2026-01-14.

B_min is found by tracing each grid point's IGRF field line with the hanetrap RK4 tracer. Lines that
leave 7 R_E are given L = 99 (far outside the belt). Grid: 2 deg latitude x 4 deg longitude x 100 km
altitude (100-2100 km); trilinear interpolation, vectorized.
"""
import math, os, sys, time
import numpy as np
from numba import njit, prange
import hane_common as F
from hanetrap import core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "Lgrid_2026014.npz")
LAT = np.arange(-90.0, 90.01, 2.0); LON = np.arange(-180.0, 180.01, 4.0); ALT = np.arange(100.0, 2100.01, 100.0)


@njit(parallel=True, cache=True)
def _bmin_batch(P, G, H, nmax, out):
    for i in prange(P.shape[0]):
        buf = np.zeros((1, 5))
        b0 = C.field_mag(P[i, 0], P[i, 1], P[i, 2], G, H, nmax)
        m = b0
        for sgn in (1.0, -1.0):
            n, closed, bhi, blo, bmin, s = C.trace_dir(P[i, 0], P[i, 1], P[i, 2], sgn, G, H, nmax, 90.0, 95.0,
                                                       0.01, 50.0, 7.0 * C.R_E, buf, False)
            if not closed:
                m = -1.0
                break
            if bmin < m:
                m = bmin
        out[i] = m


def build():
    T = F.table()
    LA, LO, AL = np.meshgrid(LAT, LON, ALT, indexing="ij")
    P = np.array([C.geodetic_to_ecef(math.radians(a), math.radians(b), h)
                  for a, b, h in zip(LA.ravel(), LO.ravel(), AL.ravel())])
    bm = np.zeros(len(P))
    t = time.time()
    _bmin_batch(P, T.G, T.H, T.nmax, bm)
    L = np.where(bm > 0, (F.B0_OH / np.where(bm > 0, bm, 1.0)) ** (1 / 3), 99.0).reshape(LA.shape)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, lat=LAT, lon=LON, alt=ALT, L=L)
    print(f"L grid {L.shape} built in {time.time()-t:.0f}s")


def load():
    d = np.load(OUT)
    return d["lat"], d["lon"], d["alt"], d["L"]


def L_of(lat, lon, alt, grid=None):
    """Vectorized trilinear L at geodetic lat, lon (deg), alt (km)."""
    la, lo, al, Lv = grid if grid is not None else load()
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    def idx(ax, v):
        v = np.clip(v, ax[0], ax[-1]); k = np.clip(np.searchsorted(ax, v) - 1, 0, len(ax) - 2)
        return k, (v - ax[k]) / (ax[k + 1] - ax[k])
    i, fi = idx(la, np.asarray(lat)); j, fj = idx(lo, lon); k, fk = idx(al, np.asarray(alt))
    out = 0.0
    for di, wi in ((0, 1 - fi), (1, fi)):
        for dj, wj in ((0, 1 - fj), (1, fj)):
            for dk, wk in ((0, 1 - fk), (1, fk)):
                out = out + wi * wj * wk * Lv[i + di, j + dj, k + dk]
    return out


if __name__ == "__main__":
    build()
