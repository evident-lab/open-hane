#!/usr/bin/env python3
"""
Orbit propagation (SGP4), TLE reading and coordinate transforms.

Running this file checks the coordinate round trip, the SGP4 wrapper, the GMST
rate, the two-body fallback and the Earth-occultation test.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional, Tuple, List
import numpy as np

try:
    from sgp4.api import Satrec, jday
    HAVE_SGP4 = True
except Exception:
    Satrec = None; jday = None; HAVE_SGP4 = False

R_E_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418

def julian_date_from_datetime(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond*1e-6)/60.0)/60.0)/24.0
    if m <= 2: y -= 1; m += 12
    A = y//100; B = 2 - A + A//4
    return math.floor(365.25*(y+4716)) + math.floor(30.6001*(m+1)) + d + B - 1524.5

def parse_datetime(dt: datetime) -> Tuple[float, float]:
    dt = dt.astimezone(timezone.utc)
    if HAVE_SGP4:
        return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond*1e-6)
    jd = julian_date_from_datetime(dt); jdi = math.floor(jd)
    return float(jdi), float(jd - jdi)

def gmst_radians(dt: datetime) -> float:
    jd, fr = parse_datetime(dt); JD = jd + fr
    T = (JD - 2451545.0)/36525.0
    gmst_sec = (67310.54841 + (876600.0*3600 + 8640184.812866)*T
                + 0.093104*T**2 - 6.2e-6*T**3)
    gmst_sec %= 86400.0
    return 2.0*math.pi*gmst_sec/86400.0

def lla_to_ecef_km(lat_deg, lon_deg, alt_km) -> np.ndarray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg); r = R_E_KM + alt_km
    return np.array([r*math.cos(la)*math.cos(lo), r*math.cos(la)*math.sin(lo), r*math.sin(la)])

def ecef_to_latlonalt_km(r_ecef_km) -> Tuple[float,float,float]:
    x, y, z = r_ecef_km; r = float(np.linalg.norm(r_ecef_km))
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z/max(r,1e-30)))))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon, r - R_E_KM

def teme_to_ecef_km(r_teme_km, dt) -> np.ndarray:
    # r_ECEF = R3(GMST) r_TEME, R3(t) = [[cos t, sin t, 0], [-sin t, cos t, 0], [0, 0, 1]]
    th = gmst_radians(dt); c, s = math.cos(th), math.sin(th)
    R = np.array([[c,s,0.0],[-s,c,0.0],[0.0,0.0,1.0]])
    return R @ r_teme_km

def sat_mean_alt_km(sat) -> float:
    n_rad_s = sat.no_kozai/60.0
    if n_rad_s <= 0: return float("nan")
    a_km = (MU_EARTH_KM3_S2/(n_rad_s*n_rad_s))**(1.0/3.0)
    return a_km - R_E_KM

def propagate_teme_km(sat, dt) -> Optional[np.ndarray]:
    jd, fr = parse_datetime(dt)
    e, r_km, _v = sat.sgp4(jd, fr)
    if e != 0: return None
    return np.array(r_km, float)

def is_occulted_segment(r0_km, r1_km, R_block_km=R_E_KM) -> bool:
    d = r1_km - r0_km; dd = float(np.dot(d, d))
    if dd <= 0: return False
    t = max(0.0, min(1.0, -float(np.dot(r0_km, d))/dd))
    closest = r0_km + t*d
    return float(np.linalg.norm(closest)) < R_block_km

class SimpleSat:
    """Two-body Kepler fallback when python-sgp4 is unavailable (rough; no drag/J2)."""
    def __init__(self, l1, l2):
        parts = l2.split()
        self.inclo = math.radians(float(parts[2]))
        self.nodeo = math.radians(float(parts[3]))
        self.ecco = float('0.'+parts[4].strip())
        self.argpo = math.radians(float(parts[5]))
        self.mo = math.radians(float(parts[6]))
        self.mm_rev_day = float(parts[7])
        self.no_kozai = self.mm_rev_day*2.0*math.pi/1440.0
        yy = int(l1[18:20]); year = 2000+yy if yy < 57 else 1900+yy
        doy = float(l1[20:32])
        jan1 = datetime(year,1,1,tzinfo=timezone.utc)
        self.epoch_dt = datetime.fromtimestamp(jan1.timestamp()+(doy-1.0)*86400.0, tz=timezone.utc)
        self.epoch_jd = julian_date_from_datetime(self.epoch_dt)
    @staticmethod
    def twoline2rv(l1, l2): return SimpleSat(l1, l2)
    def sgp4(self, jd, fr):
        dt_s = ((jd+fr) - self.epoch_jd)*86400.0
        n = self.mm_rev_day*2.0*math.pi/86400.0
        a = (MU_EARTH_KM3_S2/(n*n))**(1.0/3.0); e = min(max(self.ecco,0.0),0.99)
        M = (self.mo + n*dt_s) % (2.0*math.pi); E = M
        for _ in range(20):
            E -= (E - e*math.sin(E) - M)/max(1e-12, 1 - e*math.cos(E))
        x = a*(math.cos(E)-e); y = a*math.sqrt(max(0.0,1-e*e))*math.sin(E)
        co,so = math.cos(self.nodeo),math.sin(self.nodeo)
        ci,si = math.cos(self.inclo),math.sin(self.inclo)
        cw,sw = math.cos(self.argpo),math.sin(self.argpo)
        R = np.array([[co*cw-so*sw*ci, -co*sw-so*cw*ci, so*si],
                      [so*cw+co*sw*ci, -so*sw+co*cw*ci, -co*si],
                      [sw*si, cw*si, ci]])
        return 0, tuple((R @ np.array([x,y,0.0])).tolist()), (0.0,0.0,0.0)

# --------------------------- TLE catalog reading -------------------------
def read_tles(path, max_sats=None):
    """Read a TLE file into a list of (name, sat). Accepts 3-line and 2-line entries."""
    sats = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        if i+2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            name, l1, l2 = lines[i], lines[i+1], lines[i+2]
            sats.append((name, Satrec.twoline2rv(l1,l2) if HAVE_SGP4 else SimpleSat.twoline2rv(l1,l2)))
            i += 3
        elif i+1 < len(lines) and lines[i].startswith("1 ") and lines[i+1].startswith("2 "):
            name = f"SAT_{len(sats)+1:05d}"
            sats.append((name, Satrec.twoline2rv(lines[i],lines[i+1]) if HAVE_SGP4 else SimpleSat.twoline2rv(lines[i],lines[i+1])))
            i += 2
        else:
            i += 1
        if max_sats is not None and len(sats) >= max_sats:
            break
    return sats

# ------------------------------ checks ----------------------------------
if __name__ == "__main__":
    from datetime import timedelta
    print("Orbit and coordinate checks")
    print(f"SGP4 available: {HAVE_SGP4}")

    # standard ISS test TLE
    L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # (1) coordinate round-trip
    print("\n[1] lla -> ecef -> lla round-trip:")
    maxerr = 0
    for lat,lon,alt in [(0,0,400),(45,-75,550),(-30,150,800),(80.7,-72.7,485)]:
        r = lla_to_ecef_km(lat,lon,alt); la,lo,al = ecef_to_latlonalt_km(r)
        e = max(abs(la-lat),abs(((lo-lon+180)%360)-180),abs(al-alt)); maxerr=max(maxerr,e)
    print(f"    max error = {maxerr*1e3:.3e} (deg/km scaled) -> {'OK' if maxerr<1e-6 else 'CHECK'}")

    if HAVE_SGP4:
        sat = Satrec.twoline2rv(L1, L2)
        t0 = sat.epoch_dt if hasattr(sat,'epoch_dt') else datetime(2008,9,20,12,26,40,tzinfo=timezone.utc)
        # (2) propagate_teme matches direct sgp4
        print("\n[2] propagate_teme_km vs direct sgp4:")
        dt = datetime(2008,9,20,13,0,0,tzinfo=timezone.utc)
        jd,fr = parse_datetime(dt)
        e_direct, r_direct, _ = sat.sgp4(jd, fr)
        r_ours = propagate_teme_km(sat, dt)
        diff = float(np.linalg.norm(r_ours - np.array(r_direct)))
        print(f"    position diff = {diff:.3e} km -> {'OK' if diff<1e-9 else 'CHECK'}")

        # physical sanity: ISS altitude ~400 km
        r_ecef = teme_to_ecef_km(r_ours, dt)
        la,lo,al = ecef_to_latlonalt_km(r_ecef)
        print(f"    ISS modeled alt = {al:.1f} km (expect ~400), lat={la:.1f}, lon={lo:.1f}")

        # (4) SimpleSat fallback vs SGP4 mean altitude
        ss = SimpleSat.twoline2rv(L1, L2)
        print(f"\n[4] SimpleSat fallback mean alt={sat_mean_alt_km(ss):.1f} km "
              f"vs SGP4 mean alt={sat_mean_alt_km(sat):.1f} km (rough fallback)")

    # (3) GMST sidereal rate: net advance per solar day mod 360 = 0.9856 deg
    #     (sidereal excess; GMST gains 360.9856 deg/day, 360 of which wraps)
    print("\n[3] GMST sidereal rate:")
    d0 = datetime(2026,1,1,0,0,0,tzinfo=timezone.utc)
    g0 = gmst_radians(d0); g1 = gmst_radians(d0+timedelta(days=1))
    adv = math.degrees((g1-g0) % (2*math.pi))
    print(f"    GMST net advance/solar day (mod 360) = {adv:.4f} deg "
          f"(expect 0.9856 sidereal excess) -> {'OK' if abs(adv-0.9856)<0.01 else 'CHECK'}")

    # (5) occultation
    print("\n[5] occultation:")
    a = lla_to_ecef_km(0,0,500); b = lla_to_ecef_km(0,180,500)  # opposite sides
    c = lla_to_ecef_km(0,0,500); d = lla_to_ecef_km(10,5,500)   # same side
    print(f"    opposite sides (through Earth) occulted: {is_occulted_segment(a,b)} (expect True)")
    print(f"    same side occulted: {is_occulted_segment(c,d)} (expect False)")