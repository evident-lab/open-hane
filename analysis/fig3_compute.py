"""Fig 3 data: peak belt dose rate (1.29 betas x <S>) versus burst latitude and longitude.

1 Mt (50% fission), 500 km burst altitude, IGRF-14 at 2026-01-14, 2 mm Al. For an instantaneous
injection the belt dose rate is largest at t = 0 (afterwards electrons only diffuse and are lost),
so the peak is the maximum over L of the initial dose rate. 192 cells per burst, L smoothing 0.01
(within ~10% of the 1,536-cell converged peak).
Writes cache/fig3_peakmap.npz.
"""
import os, sys, time
import numpy as np
import hane_common as F

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "fig3_peakmap.npz")
LATS = np.arange(-65.0, 65.01, 2.5)
LONS = np.arange(-180.0, 180.01, 5.0)
ALT = 500.0

def merge(nparts=2):
    """Combine the per-process files into cache/fig3_peakmap.npz."""
    parts = [np.load(OUT.replace(".npz", f"_part{p}.npz")) for p in range(nparts)]
    out = {k: parts[0][k] for k in parts[0].files}
    for key in ("peak", "meanS", "Lburst"):
        a = parts[0][key].copy()
        for d in parts[1:]:
            a = np.where(np.isfinite(d[key]), d[key], a)
        out[key] = a
    np.savez_compressed(OUT, **out)


if __name__ == "__main__":
    part, nparts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    OUT = OUT.replace(".npz", f"_part{part}.npz") if nparts > 1 else OUT
    peak = np.full((len(LATS), len(LONS)), np.nan)
    meanS = np.full_like(peak, np.nan); Lb = np.full_like(peak, np.nan)
    t0 = time.time()
    for i, la in enumerate(LATS):
        if i % nparts != part:
            continue
        for j, lo in enumerate(LONS):
            if j == len(LONS) - 1:            # -180 and +180 are the same meridian
                peak[i, j], meanS[i, j], Lb[i, j] = peak[i, 0], meanS[i, 0], Lb[i, 0]; continue
            c = F.region_cells(float(la), float(lo), ALT, n_cells=192, fast=True)
            dr = F.dose_rate_initial(F.L_weights(c, smooth=0.01), F.BETAS["mid"])
            peak[i, j] = dr.max(); meanS[i, j] = c["meanS"]; Lb[i, j] = c["L_burst"]
        print(f"lat {la:+.0f} done ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(OUT, LATS=LATS, LONS=LONS, peak=peak, meanS=meanS, Lburst=Lb, alt=ALT,
                            beta=F.BETAS["mid"], epoch=F.EPOCH)
    print("done")
