"""IGRF-1960 drift loss cone at 120 km vs Cunningham & Cowee 2017 Fig. 1.

LANL values (read from the figure, +-2 deg): 90 deg at L ~ 1.12, ~62.5 at 1.2,
~50 at 1.3, ~43 at 1.4, ~37.3 at 1.5, ~32.5 at 1.6, ~26.5 at 1.8.
Also checks the trapping function against LANL's point-source map statements
(LA-UR-16-22015 slide 20): zero outside a South Atlantic wedge at 500 km,
roughly 0.8-0.9 over most longitudes at 1,833 km.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hanetrap import core as C, shells  # noqa: E402

EPOCH = 1960.0 + 190 / 366.0   # 1960-07-09, the epoch of the old comparison figure

LANL = {1.2: 62.5, 1.3: 50.0, 1.4: 43.0, 1.5: 37.3, 1.6: 32.5, 1.8: 26.5}


def run():
    T = shells.ShellTable(EPOCH)
    L = np.array([1.10, 1.11, 1.12, 1.13, 1.15] + list(LANL))
    a = T.dlc_angle_deg(L, hc=120)
    print("L     ours(deg)  LANL(deg)")
    for Li, ai in zip(L, a):
        print(f"{Li:4.2f}  {ai:7.2f}   {LANL.get(round(Li, 2), '')}")
    # first L where the cone opens below 90 deg
    Ls = np.arange(1.08, 1.2, 0.002)
    aa = T.dlc_angle_deg(Ls, hc=120)
    open_L = Ls[np.argmax(aa < 89.9)]
    print("cone opens below 90 deg at L =", round(float(open_L), 3), "(LANL ~1.12)")
    worst = max(abs(T.dlc_angle_deg(np.array([k]), 120)[0] - v) for k, v in LANL.items())
    print("max |difference| over LANL points:", round(worst, 2), "deg")
    print("SAA longitude of shell minimum at L=1.2 (deg E):",
          round(math.degrees(T.lons[T.argmin_lon[120][np.searchsorted(T.Bgrid, T.B0 / 1.2**3)]]), 1))
    # point-source maps
    for h in (500.0, 1833.0):
        lats = np.arange(-60, 61, 10.0); lons = np.arange(-180, 180, 15.0)
        pts = np.array([C.geodetic_to_ecef(math.radians(la), math.radians(lo), h) for la in lats for lo in lons])
        S, _, _ = shells.S_points(T, pts, hc=120)
        S = S.reshape(len(lats), len(lons))
        nz = lons[np.any(S > 0.05, axis=0)]
        print(f"h={h:.0f} km: median S={np.median(S):.2f}; longitudes with S>0.05: {nz.min() if len(nz) else None}..{nz.max() if len(nz) else None}")
    return T, worst, open_L


if __name__ == "__main__":
    run()
