"""IGRF field values, geodesy round trip and field-line tracing in an exact dipole."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hanetrap import core as C  # noqa: E402


def test_igrf_against_ppigrf():
    import ppigrf
    from datetime import datetime
    G, H, nmax = C.load_igrf(1960.0 + 190 / 366.0)
    rng = np.random.default_rng(0)
    r = 6371.2 + rng.uniform(80, 20000, 200)
    th = rng.uniform(1, 179, 200)
    ph = rng.uniform(-180, 180, 200)
    Br, Bt, Bp = ppigrf.igrf_gc(r, th, ph, datetime(1960, 7, 9))
    Br, Bt, Bp = (np.asarray(a).ravel() for a in (Br, Bt, Bp))
    worst = 0.0
    for i in range(200):
        t, p = math.radians(th[i]), math.radians(ph[i])
        x, y, z = r[i] * math.sin(t) * math.cos(p), r[i] * math.sin(t) * math.sin(p), r[i] * math.cos(t)
        bx, by, bz = C.field_ecef(x, y, z, G, H, nmax)
        br = (bx * x + by * y + bz * z) / r[i]
        bt = bx * math.cos(t) * math.cos(p) + by * math.cos(t) * math.sin(p) - bz * math.sin(t)
        bp = -bx * math.sin(p) + by * math.cos(p)
        worst = max(worst, abs(br - Br[i]), abs(bt - Bt[i]), abs(bp - Bp[i]))
    assert worst < 1.0, worst   # within 1 nT; in practice agreement is at rounding level
    return worst


def test_geodesy_roundtrip():
    for lat in (-80, -30, 0, 16.7, 45, 89):
        for h in (0.0, 100.0, 400.0, 2000.0, 20000.0):
            x, y, z = C.geodetic_to_ecef(math.radians(lat), math.radians(190.5), h)
            assert abs(C.geodetic_alt(x, y, z) - h) < 1e-3
            assert abs(math.degrees(C.geodetic_lat(x, y, z)) - lat) < 1e-5


def test_dipole_trace():
    # Pure axial dipole, g10 = -30000 nT: B = B0/L^3 sqrt(1 + 3 sin^2) / cos^6
    G = np.zeros((2, 2)); H = np.zeros((2, 2)); G[1, 0] = -30000.0
    buf = np.zeros((1, 5))
    for L in (1.2, 2.0, 4.0):
        x0 = L * C.A_IGRF
        closed, bmin, blo, bhi, b0 = C.line_summary(x0, 0.0, 0.0, G, H, 1, 100.0, 120.0,
                                                     0.004, 25.0, 20 * C.R_E, buf)
        assert closed
        assert abs(bmin - 30000.0 / L ** 3) / (30000.0 / L ** 3) < 1e-6
    return True


if __name__ == "__main__":
    print("IGRF max |dB| vs ppigrf (nT):", test_igrf_against_ppigrf())
    test_geodesy_roundtrip(); print("geodesy round trip OK")
    test_dipole_trace(); print("dipole trace OK")
