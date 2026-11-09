"""Atmosphere, beta source, stopping and scattering.

Sources, with printed pages:
- Neutral species 100-10,000 km: TRH Table 11-7 (printed p. 11-24; Anderson &
  Francis model, average solar activity, 45 deg, diurnal average), transcribed
  from the page image. One O2 entry at 500 km is illegible in the scan and is
  log-interpolated between 400 and 600 km (negligible: O2 is <0.02% there).
- Below 100 km: U.S. Standard Atmosphere 1976 mass density (data file) with
  the well-mixed composition N2 78.084%, O2 20.9476%, Ar 0.934% (TRH p. 11-9).
- LANL ionized-air profile for the burst region: LA-UR-12-23076 slide 12.
- Beta rate: DNA 4501F Eq. 3-10, printed p. 166.
- Electronic stopping: Lindhard-Scharff, DNA 4501F Eq. 4-46 (p. 213), in SI.
- Nuclear stopping: ZBL universal (Ziegler, Biersack & Littmark 1985).
- Multiple scattering: screened Rutherford, small angle.
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import njit

from . import core as C

# ------------------------------------------------------------------ TRH Table 11-7
_H = np.array([100, 120, 140, 160, 180, 200, 240, 300, 340, 400, 500, 600, 700, 800,
               900, 1000, 1400, 2000, 4000, 6000, 10000], float)
_TAB = {  # cm^-3
    "H":  [7.04e5, 6.00e4, 3.70e4, 2.59e4, 2.18e4, 1.93e4, 1.61e4, 1.37e4, 1.29e4, 1.21e4, 1.10e4,
           9.98e3, 9.09e3, 8.31e3, 7.61e3, 6.98e3, 5.07e3, 3.32e3, 1.14e3, 5.42e2, 2.03e2],
    "He": [1.38e8, 5.58e7, 2.88e7, 1.79e7, 1.37e7, 1.11e7, 7.98e6, 5.59e6, 4.66e6, 3.65e6, 2.47e6,
           1.69e6, 1.17e6, 8.14e5, 5.74e5, 4.09e5, 1.14e5, 2.13e4, 3.20e2, 1.86e1, 0],
    "O":  [7.65e11, 1.85e11, 4.67e10, 1.79e10, 9.34e9, 5.42e9, 2.13e9, 6.79e8, 3.46e8, 1.32e8,
           2.75e7, 6.01e6, 1.37e6, 3.25e5, 8.04e4, 2.06e4, 1.27e2, 0, 0, 0, 0],
    "N2": [8.83e12, 4.64e11, 5.76e10, 1.37e10, 4.86e9, 2.01e9, 4.34e8, 6.28e7, 1.96e7, 3.62e6,
           2.33e5, 1.63e4, 1.22e3, 9.88e1, 8.56, 0, 0, 0, 0, 0, 0],
    "O2": [1.89e12, 6.81e10, 6.66e9, 1.35e9, 4.22e8, 1.56e8, 2.75e7, 3.06e6, 8.10e5, 1.18e5,
           math.nan, 2.44e2, 1.27e1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Ar": [7.83e10, 1.78e9, 1.08e8, 1.59e7, 3.86e6, 1.14e6, 1.34e5, 8.86e3, 1.68e3, 1.51e2, 3.01,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}
_TAB["O2"][10] = math.exp(0.5 * (math.log(1.18e5) + math.log(2.44e2)))  # illegible entry

# target atoms: (name, Z2, M2 in u)
TARGETS = (("H", 1, 1.008), ("He", 2, 4.003), ("N", 7, 14.007), ("O", 8, 15.999), ("Ar", 18, 39.95))
Z2 = np.array([t[1] for t in TARGETS], float)
M2 = np.array([t[2] for t in TARGETS], float)

DZ = 1.0  # km grid spacing
ZMAX = 10000.0


def _ussa76():
    z, rho = [], []
    with open(os.path.join(C.DATA, "US_Standard_Atm_1976.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 7:
                try:
                    a, b = float(p[0]), float(p[6])
                except ValueError:
                    continue
                z.append(a); rho.append(b)
    return np.array(z), np.array(rho)


def build_atom_table():
    """Atom number densities (m^-3) on a 1-km altitude grid, columns = TARGETS."""
    zg = np.arange(0.0, ZMAX + DZ, DZ)
    out = np.zeros((len(zg), len(TARGETS)))
    floor = 1e-30
    logs = {k: np.log(np.maximum(np.array(v, float), floor)) for k, v in _TAB.items()}
    hi = zg >= 100.0
    for k in logs:
        val = np.exp(np.interp(zg[hi], _H, logs[k])) * 1e6   # cm^-3 -> m^-3
        val[val < 1e-20] = 0.0
        logs[k] = val
    out[hi, 0] = logs["H"]
    out[hi, 1] = logs["He"]
    out[hi, 2] = 2 * logs["N2"]
    out[hi, 3] = logs["O"] + 2 * logs["O2"]
    out[hi, 4] = logs["Ar"]
    z76, rho76 = _ussa76()
    lo = ~hi
    rho = np.exp(np.interp(zg[lo], z76, np.log(rho76)))
    n = rho / (28.9644 * C.AMU)                        # molecules m^-3, well mixed
    out[lo, 2] = 2 * 0.78084 * n
    out[lo, 3] = 2 * 0.209476 * n
    out[lo, 4] = 0.00934 * n
    return zg, out


ATOMS_Z, ATOMS = build_atom_table()


@njit(cache=True)
def atoms_at(alt, table):
    """Linear interpolation of the atom table at geodetic altitude (km)."""
    if alt < 0.0:
        alt = 0.0
    if alt >= ZMAX:
        return table[-1]
    i = int(alt / DZ)
    u = alt / DZ - i
    return table[i] * (1 - u) + table[i + 1] * u


@njit(cache=True)
def rho_ion_lanl(alt, maglat):
    """LANL ionized-air mass density (kg/m^3), LA-UR-12-23076 slide 12."""
    rho400 = 8e-15  # 8e-18 g/cm^3
    if alt >= 320.0:
        lr = -(alt - 400.0) / 100.0
    elif alt >= 180.0:
        lr = 0.8 - (alt - 320.0) / 20.0
    else:
        lr = 7.8 - (alt - 180.0) / 10.0
    lam = min(abs(maglat), math.radians(85.0))
    return rho400 * math.exp(lr) / math.cos(lam) ** 2


# ------------------------------------------------------------------ beta source
@njit(cache=True)
def beta_rate_fission(t):
    """betas per fission per second, U-235 (DNA 4501F Eq. 3-10)."""
    if t < 0.5:
        return 0.33
    if t < 1000.0:
        return 0.5 / (1.0 + t)
    return 2.0 * t ** -1.2


@njit(cache=True)
def _piece_int(k, a, b):
    if k == 0:
        return 0.33 * (b - a)
    if k == 1:
        return 0.5 * math.log((1.0 + b) / (1.0 + a))
    return 10.0 * (a ** -0.2 - b ** -0.2)


@njit(cache=True)
def beta_integral(t1, t2):
    """betas per fission emitted in [t1, t2]."""
    edges = (0.0, 0.5, 1000.0, 1e300)
    tot = 0.0
    for k in range(3):
        a = max(t1, edges[k]); b = min(t2, edges[k + 1])
        if b > a:
            tot += _piece_int(k, a, b)
    return tot


@njit(cache=True)
def beta_sample_time(t1, t2, u):
    """emission time within [t1, t2], drawn in proportion to the rate."""
    edges = (0.0, 0.5, 1000.0, 1e300)
    tot = beta_integral(t1, t2)
    target = u * tot
    acc = 0.0
    for k in range(3):
        a = max(t1, edges[k]); b = min(t2, edges[k + 1])
        if b <= a:
            continue
        w = _piece_int(k, a, b)
        if acc + w >= target or k == 2:
            f = (target - acc) / w if w > 0 else 0.0
            if k == 0:
                return a + f * (b - a)
            if k == 1:
                return (1.0 + a) * ((1.0 + b) / (1.0 + a)) ** f - 1.0
            return (a ** -0.2 - f * (a ** -0.2 - b ** -0.2)) ** -5.0
        acc += w
    return t2


# ------------------------------------------------------------------ stopping
@njit(cache=True)
def S_elec(Z1, Z2_, v):
    """Lindhard-Scharff electronic stopping cross-section, J m^2 per atom."""
    return (8.0 * math.pi * C.KE * C.E_CHARGE ** 2 * C.A0
            * Z1 ** (7.0 / 6.0) * Z2_ / (Z1 ** (2.0 / 3.0) + Z2_ ** (2.0 / 3.0)) ** 1.5 * v / C.V0)


@njit(cache=True)
def S_nucl(Z1, M1, Z2_, M2_, E_keV):
    """ZBL universal nuclear stopping cross-section, J m^2 per atom."""
    zs = Z1 ** 0.23 + Z2_ ** 0.23
    eps = 32.53 * M2_ * E_keV / (Z1 * Z2_ * (M1 + M2_) * zs)
    if eps <= 30.0:
        sn = math.log(1.0 + 1.1383 * eps) / (2.0 * (eps + 0.01321 * eps ** 0.21226 + 0.19593 * eps ** 0.5))
    else:
        sn = math.log(eps) / (2.0 * eps)
    ev_cm2 = 8.462e-15 * Z1 * Z2_ * M1 * sn / ((M1 + M2_) * zs)
    return ev_cm2 * C.E_CHARGE * 1e-4


@njit(cache=True)
def dEds(Z1, M1, E_J, n_atoms, Z2a, M2a):
    """energy loss per metre (J/m) summed over target species (Bragg additivity)."""
    v = math.sqrt(2.0 * E_J / (M1 * C.AMU))
    E_keV = E_J / C.E_CHARGE / 1e3
    tot = 0.0
    for k in range(n_atoms.shape[0]):
        if n_atoms[k] > 0.0:
            tot += n_atoms[k] * (S_elec(Z1, Z2a[k], v) + S_nucl(Z1, M1, Z2a[k], M2a[k], E_keV))
    return tot


@njit(cache=True)
def theta2_per_m(Z1, M1, E_J, n_atoms, Z2a, M2a):
    """mean-square space scattering angle per metre (rad^2/m).

    Small-angle screened Rutherford: 2 pi n (2 Z1 Z2 e^2 / (4 pi eps0 p v))^2 ln(th_max/th_min),
    th_min with the ZBL screening length, th_max = M2/M1. Where single collisions are
    no longer small-angle (th_min > th_max / e), the logarithm is floored at 1:
    this floor is our own choice and marks the edge of validity.
    """
    m = M1 * C.AMU
    v = math.sqrt(2.0 * E_J / m)
    pv = m * v * v
    tot = 0.0
    for k in range(n_atoms.shape[0]):
        if n_atoms[k] <= 0.0:
            continue
        b = 2.0 * Z1 * Z2a[k] * C.KE * C.E_CHARGE ** 2 / pv
        aU = 0.8854 * C.A0 / (Z1 ** 0.23 + Z2a[k] ** 0.23)
        thmin = b / aU
        thmax = M2a[k] / M1 if M2a[k] < M1 else 1.0
        lg = math.log(thmax / thmin)
        if lg < 1.0:
            lg = 1.0
        tot += 2.0 * math.pi * n_atoms[k] * b * b * lg
    return tot
