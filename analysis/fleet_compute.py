"""Fleet response to one burst: per-satellite prompt X-ray dose and cumulative belt dose.

    python fleet_compute.py LAT LON ALT TAG      e.g.  python fleet_compute.py -20 -35 500 saa

Writes cache/fleet_<TAG>.npz with, for every LEO object (0-2000 km at the burst time):
    prompt X-ray fluence after air (J/m^2) and dose (krad(Si)) at t = 0;
    cumulative belt dose (krad(Si)) every 3 h for 60 days, for the three trapping cases
    (beta = 0.404, 1.289, 5.207 betas per fission);
    position at t = 0, inclination, perigee and apogee altitude.
Dose-limit curves for any limit are then derived without recomputation (fig4 render, tables).

HANE_DAYS sets the run length (default 60) and HANE_YIELD_KT the yield (default 1000).

Method (equations in docs/open_hane_methods.pdf):
  - trapped source: IGRF expansion region with full-drift-shell <S>, 1,536 cells;
  - belt: Carter spectrum, Brautigam-Albert D_LL at Kp = 2, empirical lifetimes,
    dipole drift-shell volumes above 80 km, 3-hour steps, 2 mm Al dose response;
  - satellites: SGP4 every 20 min, TEME->ECEF r = R3(GMST) r_TEME, L from the IGRF-14 lookup grid;
    local density factor f = S(x)/Sbar(L).
"""
import math, os, sys, time
from datetime import datetime, timezone, timedelta
import numpy as np
import hane_common as F
import lgrid
import tier2_orbits as T2O
import tier2_prompt_xray as T2X
from sgp4.api import SatrecArray

HERE = os.path.dirname(os.path.abspath(__file__))
TLE = F.TLE_FILE
TREF = datetime(2026, 1, 14, 8, 0, 0, tzinfo=timezone.utc)
DAYS = float(os.environ.get("HANE_DAYS", 60.0))   # run length (days); default 60
DT_BELT = 3 * 3600.0
DT_ORB = 20 * 60.0
FLUENCE_KILL = 40.0      # J/m^2 after air, prompt failure threshold
MU = 398600.4418         # km^3/s^2
RE = 6378.137            # km, SGP4 (WGS-72) equatorial radius for perigee/apogee altitudes


def teme_to_ecef(r, when):
    th = T2O.gmst_radians(when); c, s = math.cos(th), math.sin(th)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


def ecef_to_lla(r):
    rr = np.linalg.norm(r, axis=-1)
    return (np.degrees(np.arcsin(np.clip(r[..., 2] / rr, -1, 1))),
            np.degrees(np.arctan2(r[..., 1], r[..., 0])), rr - 6371.0)


def fleet(lat, lon, alt, tag):
    t0 = time.time()
    sats = [s for _, s in T2O.read_tles(TLE)]
    arr = SatrecArray(sats)
    jd0, fr0 = T2O.parse_datetime(TREF)
    e, r, _ = arr.sgp4(np.array([jd0]), np.array([fr0]))
    r0 = teme_to_ecef(r[:, 0, :], TREF); la0, lo0, al0 = ecef_to_lla(r0)
    idx = np.nonzero((e[:, 0] == 0) & (al0 > 0) & (al0 <= 2000.0))[0]
    sel = [sats[i] for i in idx]; arr = SatrecArray(sel)
    r0, la0, lo0, al0 = r0[idx], la0[idx], lo0[idx], al0[idx]; n = len(idx)
    inc = np.degrees([s.inclo for s in sel])
    a = np.array([(MU / (s.no_kozai / 60.0) ** 2) ** (1 / 3) for s in sel])
    ecc = np.array([s.ecco for s in sel])
    perigee, apogee = a * (1 - ecc) - RE, a * (1 + ecc) - RE
    print(f"[{tag}] {n} LEO objects", flush=True)

    rb = T2X.lla_to_ecef_km(lat, lon, alt)
    sc = T2X.PromptScenario(yield_kt=F.YIELD_KT, shield_mm_al=F.SHIELD)
    pr = np.array([T2X.prompt_xray_dose_krad_si(rb, x, sc)[:2] for x in r0])
    dprompt, Fp = pr[:, 0], pr[:, 1]

    cells = F.region_cells(lat, lon, alt, n_cells=1536)
    Lw = F.L_weights(cells, smooth=0.005)
    tb = np.arange(0.0, DAYS * 86400.0 + 1, DT_BELT)
    belts = {m: F.evolve(Lw, m, tb) for m in ("low", "mid", "high")}
    print(f"[{tag}] <S> = {cells['meanS']:.4f}; peak rad/s " +
          ", ".join(f"{m} {belts[m].max():.2f}" for m in belts) + f" ({time.time()-t0:.0f}s)", flush=True)

    grid = lgrid.load(); SBAR = F.shell_Sbar(F.LG)
    nout = len(tb)
    dose = {m: np.zeros(n) for m in belts}
    cum = {m: np.zeros((n, nout), np.float32) for m in belts}
    for k, ts in enumerate(np.arange(0.0, DAYS * 86400.0, DT_ORB)):
        when = TREF + timedelta(seconds=float(ts)); jd, fr = T2O.parse_datetime(when)
        e, r, _ = arr.sgp4(np.array([jd]), np.array([fr]))
        rec = teme_to_ecef(r[:, 0, :], when); la, lo, al = ecef_to_lla(rec)
        Ls = np.where(e[:, 0] == 0, lgrid.L_of(la, lo, np.clip(al, 100, 2100), grid), 99.0)
        inb = (Ls >= F.LG[0]) & (Ls <= F.LG[-1]); fac = np.zeros(n)
        if inb.any():
            sb = np.interp(Ls[inb], F.LG, SBAR); sl = F.local_S(rec[inb], Ls[inb])
            fac[inb] = np.where(sb > 0, sl / np.where(sb > 0, sb, 1.0), 0.0)
        ib = min(int(ts // DT_BELT), nout - 1)
        for m, dr in belts.items():
            dose[m] += np.interp(Ls, F.LG, dr[ib], left=0.0, right=0.0) * fac * DT_ORB / 1000.0
        if (k + 1) % int(DT_BELT // DT_ORB) == 0:
            for m in belts:
                cum[m][:, min(ib + 1, nout - 1)] = dose[m]
        if k % 1080 == 0:
            print(f"[{tag}]   day {ts/86400:.0f} ({time.time()-t0:.0f}s)", flush=True)
    for m in belts:
        np.maximum.accumulate(cum[m], axis=1, out=cum[m])
    out = os.path.join(HERE, "cache", f"fleet_{tag}.npz")
    np.savez_compressed(out, times=tb, Fp=Fp, dprompt=dprompt, cum_low=cum["low"], cum_mid=cum["mid"],
                        cum_high=cum["high"], lat=la0, lon=lo0, alt=al0, inc=inc, perigee=perigee, apogee=apogee,
                        burst=np.array([lat, lon, alt]), meanS=cells["meanS"],
                        cell_L=cells["L"], cell_w=cells["w"], cell_S=cells["S"], LG=F.LG,
                        belt_low=belts["low"].astype(np.float32), belt_mid=belts["mid"].astype(np.float32),
                        belt_high=belts["high"].astype(np.float32), fluence_kill=FLUENCE_KILL,
                        burst_L=float(np.nanmean(cells["L"])))
    print(f"[{tag}] wrote {out} ({time.time()-t0:.0f}s)", flush=True)


def loss_curve(d, case, krad):
    """Percent of the fleet lost versus time for a dose limit (prompt fluence or prompt + belt dose)."""
    dead = (d["Fp"][:, None] >= float(d["fluence_kill"])) | (d["dprompt"][:, None] + d[f"cum_{case}"] >= krad)
    return 100.0 * dead.mean(axis=0)


if __name__ == "__main__":
    la, lo, al = map(float, sys.argv[1:4]); tag = sys.argv[4]
    fleet(la, lo, al, tag)
