#!/usr/bin/env python3
"""
Fission-electron source, debris burst region and trapped-electron flux.

Constants, the Carter beta spectrum, debris mass and confinement radius, and the
dipole drift-shell volume used to turn a trapped electron count into a flux.
Running this file prints a set of reference checks.

Units: km for radii and altitudes, kg, J, MeV, tesla.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ----------------------------- constants ---------------------------------
R_E_KM = 6371.0
R_E_M  = 6371.0e3
B0_EQ_T = 3.12e-5
MU0 = 4.0e-7 * math.pi
C_LIGHT = 2.99792458e8
MEC2_MEV = 0.51099895
J_PER_KT = 4.184e12
MEV_TO_J = 1.602176634e-13
MEV_PER_FISSION = 180.0
FISSIONS_PER_KT = J_PER_KT / (MEV_PER_FISSION * MEV_TO_J)
BETAS_PER_FISSION_TOTAL = 6.0
WAY_WIGNER_COEFF_MEV_S_FISSION = 1.26
CARTER_MEAN_MEV = 1.358
H_ATM_KM_DEFAULT = 80.0          # lower edge of drift-shell volumes (km), as in the LANL ESM

# --------------- centred tilted dipole (reference field for the checks) ---
def _unit_vec(lat_deg, lon_deg):
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(la)*math.cos(lo), math.cos(la)*math.sin(lo), math.sin(la)])

class CenteredDipole:
    def __init__(self, pole_lat_deg=80.7, pole_lon_deg=-72.7):
        self.zhat = _unit_vec(pole_lat_deg, pole_lon_deg)
        ez = np.array([0.0, 0.0, 1.0])
        y = np.cross(self.zhat, ez)
        if np.linalg.norm(y) < 1e-10:
            y = np.array([0.0, 1.0, 0.0])
        self.yhat = y/np.linalg.norm(y)
        self.xhat = np.cross(self.yhat, self.zhat); self.xhat /= np.linalg.norm(self.xhat)
    def L_lambda_phi(self, r_ecef_km):
        rd = np.array([np.dot(r_ecef_km, self.xhat), np.dot(r_ecef_km, self.yhat), np.dot(r_ecef_km, self.zhat)])
        rr = float(np.linalg.norm(rd))
        if rr <= 0: return float("nan"), float("nan"), float("nan"), float("nan")
        lam = math.asin(max(-1.0, min(1.0, rd[2]/rr)))
        phi = math.atan2(rd[1], rd[0])
        c = max(abs(math.cos(lam)), 1e-8)
        return (rr/R_E_KM)/(c*c), lam, phi, rr
    def L_shell(self, r_ecef_km): return self.L_lambda_phi(r_ecef_km)[0]
    @staticmethod
    def dipole_B_T(L, lam_rad):
        c = max(abs(math.cos(lam_rad)), 1e-10); s = math.sin(lam_rad)
        return (B0_EQ_T/(L**3))*math.sqrt(1.0+3.0*s*s)/(c**6)

def lla_to_ecef_km(lat_deg, lon_deg, alt_km):
    la, lo = math.radians(lat_deg), math.radians(lon_deg); r = R_E_KM + alt_km
    return np.array([r*math.cos(la)*math.cos(lo), r*math.cos(la)*math.sin(lo), r*math.sin(la)])

# ----------------------------- scenario ----------------------------------
@dataclass
class SourceScenario:
    yield_kt: float = 1000.0
    fission_fraction: float = 0.50
    debris_kinetic_fraction: float = 0.25
    debris_speed_km_s: float = 1500.0
    burst_lat_deg: float = 40.75
    burst_lon_deg: float = -35.5
    burst_alt_km: float = 485.0
    h_atm_km: float = H_ATM_KM_DEFAULT
    geomag_pole_lat_deg: float = 80.7
    geomag_pole_lon_deg: float = -72.7

# --------------------------- source term ---------------------------------
def total_fissions(sc: SourceScenario) -> float:
    return sc.yield_kt * sc.fission_fraction * FISSIONS_PER_KT

def way_wigner_betas_per_fission_cumulative(t1_s: float, t2_s: float) -> float:
    t1 = max(float(t1_s), 1e-12); t2 = max(float(t2_s), t1)
    A = WAY_WIGNER_COEFF_MEV_S_FISSION / CARTER_MEAN_MEV
    return (A/0.2)*(t1**(-0.2) - t2**(-0.2))

def carter_beta_energy_weights(E_MeV: np.ndarray) -> np.ndarray:
    """Discrete Carter spectrum weights, w_i proportional to J(E_i) dE_i, normalised to 1.

    Using bin widths (rather than J(E_i) alone) keeps the 1.358 MeV mean energy on a
    geometric energy grid.
    """
    E = np.asarray(E_MeV, dtype=float)
    g = 1.0 + E/MEC2_MEV
    beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0/(g*g)))
    J = beta*np.exp(-0.2938*(g-1.0) - 0.0144*(g-1.0)**2)   # Carter shape
    J = np.maximum(J, 0.0)
    dE = np.gradient(E)                                     # trapezoidal bin widths
    w = J*dE
    s = float(np.sum(w))
    return w/s if s > 0 else np.ones_like(E)/len(E)

def electron_speed_m_s(E_MeV: np.ndarray) -> np.ndarray:
    g = 1.0 + np.asarray(E_MeV, dtype=float)/MEC2_MEV
    return np.sqrt(np.maximum(0.0, 1.0 - 1.0/(g*g)))*C_LIGHT


def local_isotropic_survival_from_equatorial(eta_equatorial, B_local_over_B_eq):
    """Survival for locally isotropic beta emission after adiabatic mapping."""
    eta_eq = np.clip(np.asarray(eta_equatorial, float), 0.0, 1.0)
    ratio = np.maximum(np.asarray(B_local_over_B_eq, float), 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - ratio * (1.0 - eta_eq * eta_eq)))

# --------------------------- burst region --------------------------------
def ionosphere_density_kg_m3(h_km: float, mag_lat_deg: float) -> float:
    H_C, H_D = 180.0, 320.0; H1, H2, H3 = 10.0, 20.0, 100.0
    H_ANCHOR, RHO_ANCHOR = 400.0, 8e-15
    hh = max(float(h_km), 0.0)
    if hh >= H_D:   lr = -(hh-H_ANCHOR)/H3
    elif hh >= H_C: lr = -(H_D-H_ANCHOR)/H3 - (hh-H_D)/H2
    else:           lr = -(H_D-H_ANCHOR)/H3 - (H_C-H_D)/H2 - (hh-H_C)/H1
    lat_factor = math.cos(math.radians(min(abs(mag_lat_deg), 85.0)))**-2
    return RHO_ANCHOR*math.exp(lr)*lat_factor

def debris_mass_kg(sc: SourceScenario) -> Tuple[float, float]:
    E_ke_J = sc.debris_kinetic_fraction*sc.yield_kt*J_PER_KT
    u = sc.debris_speed_km_s*1000.0
    return 2.0*E_ke_J/(u*u), E_ke_J

def magnetic_confinement_radius_km(E_ke_J: float, B_T: float) -> float:
    energy_density = B_T*B_T/(2.0*MU0)          # magnetic energy density B^2 / 2 mu0
    R_m = (3.0*E_ke_J/(4.0*math.pi*max(energy_density, 1e-300)))**(1.0/3.0)
    return R_m/1000.0

def equal_mass_radius_km(M_target_kg, burst_alt_km, mag_lat_deg, direction,
                         ds_km=2.0, nth=160, max_R_km=6000.0) -> float:
    r0 = R_E_KM + burst_alt_km
    th = np.linspace(0.0, math.pi/2.0, nth); dth = th[1]-th[0]
    s = ds_km; cum = 0.0
    while s < max_R_km:
        rc = np.sqrt(r0*r0 + s*s + 2.0*r0*s*np.cos(th)*direction)
        alt = np.maximum(rc - R_E_KM, 0.0)
        rho = np.array([ionosphere_density_kg_m3(a, mag_lat_deg) for a in alt])
        dV_km3 = 2.0*math.pi*s*s*np.sin(th)*dth*ds_km
        cum += float(np.sum(rho*dV_km3*1e9))    # km^3 -> m^3
        if cum >= M_target_kg: return s
        s += ds_km
    return max_R_km

def transverse_mass_radius_km(M_target_kg, burst_alt_km, mag_lat_deg,
                              ds_km=2.0, nth=160, max_R_km=6000.0) -> float:
    """Equal-mass radius in the horizontal direction.

    The swept sphere stays near the burst altitude, so the density is taken at the
    burst's altitude band. Used for comparison with the ESM transverse extent only;
    the L range comes from the upward and downward radii.
    """
    r0 = R_E_KM + burst_alt_km
    th = np.linspace(0.0, math.pi/2.0, nth); dth = th[1]-th[0]
    s = ds_km; cum = 0.0
    while s < max_R_km:
        alt = max(math.sqrt(r0*r0 + s*s) - R_E_KM, 0.0)
        rho = ionosphere_density_kg_m3(alt, mag_lat_deg)
        dV_km3 = 2.0*math.pi*s*s*float(np.sum(np.sin(th)*dth))*ds_km
        cum += rho*dV_km3*1e9
        if cum >= M_target_kg: return s
        s += ds_km
    return max_R_km

def compute_burst_region(sc: SourceScenario, model) -> Dict[str, float]:
    M_d, E_ke_J = debris_mass_kg(sc)
    r_burst = lla_to_ecef_km(sc.burst_lat_deg, sc.burst_lon_deg, sc.burst_alt_km)
    L_b, lam_b, _phi, _rr = model.L_lambda_phi(r_burst)
    # use the model's own |B| when it has one (IGRF), otherwise the dipole formula
    if hasattr(model, "field_magnitude_T"):
        B_b = model.field_magnitude_T(r_burst)
    else:
        B_b = model.dipole_B_T(L_b, lam_b)
    Rb_km = magnetic_confinement_radius_km(E_ke_J/2.0, B_b)
    mag_lat_deg = math.degrees(lam_b)
    Rm_up = equal_mass_radius_km(M_d/2.0, sc.burst_alt_km, mag_lat_deg, +1)
    Rm_dn = equal_mass_radius_km(M_d/2.0, sc.burst_alt_km, mag_lat_deg, -1)
    Rm_tr = transverse_mass_radius_km(M_d/2.0, sc.burst_alt_km, mag_lat_deg)
    R_up, R_dn, R_tr = min(Rm_up, Rb_km), min(Rm_dn, Rb_km), min(Rm_tr, Rb_km)
    alt_top = sc.burst_alt_km + R_up; alt_bot = max(0.0, sc.burst_alt_km - R_dn)
    L_top = model.L_shell(lla_to_ecef_km(sc.burst_lat_deg, sc.burst_lon_deg, alt_top))
    L_bot = model.L_shell(lla_to_ecef_km(sc.burst_lat_deg, sc.burst_lon_deg, alt_bot))
    Lmin, Lmax = sorted([L_top, L_bot])
    return dict(M_debris_kg=M_d, E_debris_KE_J=E_ke_J, L_burst=L_b, mag_lat_deg=mag_lat_deg,
                B_burst_T=B_b, R_magnetic_confinement_km=Rb_km,
                R_equal_mass_up_km=Rm_up, R_equal_mass_down_km=Rm_dn,
                R_equal_mass_trans_km=Rm_tr,
                R_up_km=R_up, R_down_km=R_dn, R_trans_km=R_tr,
                L_source_min=Lmin, L_source_max=Lmax)

# --------------------------- drift volume & flux -------------------------
def drift_shell_volume_bin_m3(L1, L2, h_atm_km=H_ATM_KM_DEFAULT, nL=8, nlam=80) -> float:
    # dipole flux-tube volume: integrand L^2 cos^7(lambda)
    Ls = np.linspace(L1, L2, nL); total = 0.0
    for j in range(len(Ls)-1):
        L = 0.5*(Ls[j]+Ls[j+1]); dL = Ls[j+1]-Ls[j]
        arg = (R_E_KM + h_atm_km)/(L*R_E_KM)
        if arg >= 1.0: continue
        lam_m = math.acos(math.sqrt(max(0.0, arg)))
        lam = np.linspace(-lam_m, lam_m, nlam)
        integ = (R_E_M**3)*L*L*np.cos(lam)**7
        total += float(trapz(integ, lam)*dL*2.0*math.pi)
    return max(total, 1.0)

def assemble_flux(N_trapped, L1, L2, h_atm_km=H_ATM_KM_DEFAULT,
                  Emin_MeV=0.05, Emax_MeV=8.0, nE=64) -> Dict[str, float]:
    """Omnidirectional flux phi = n v from a trapped electron count spread over an L band."""
    V = drift_shell_volume_bin_m3(L1, L2, h_atm_km)
    E = np.geomspace(Emin_MeV, Emax_MeV, nE)
    w = carter_beta_energy_weights(E)          # discrete, sum=1
    v = electron_speed_m_s(E)
    n = N_trapped/V
    phi_cm2_s = float(np.sum(n*w*v))/1.0e4      # /m^2/s -> /cm^2/s
    return dict(V_m3=V, n_density_m3=n, phi_total_cm2s=phi_cm2_s,
                mean_E_MeV=float(np.sum(E*w)))

# ------------------------------ reference checks -------------------------
if __name__ == "__main__":
    print("Source, burst region and flux: reference values")
    sc = SourceScenario()
    model = CenteredDipole(sc.geomag_pole_lat_deg, sc.geomag_pole_lon_deg)

    print(f"\n[fissions/kt]      {FISSIONS_PER_KT:.3e}   (target 1.45e23)")
    print(f"[Way-Wigner 1-1e5] {way_wigner_betas_per_fission_cumulative(1,1e5):.4f}   (target ~4.18)")
    md, eke = debris_mass_kg(sc)
    print(f"[debris mass]      {md:.1f} kg   (target 929.8)")

    E = np.geomspace(0.05, 8.0, 64); w = carter_beta_energy_weights(E)
    print(f"[Carter mean]      {np.sum(E*w):.3f} MeV   (target 1.358)   sum(w)={np.sum(w):.4f}")

    print("\n[burst region] equatorial 1 Mt burst, centred dipole:")
    for alt, ref in [(400.0, (464.6,165.9,320.2)), (600.0, (915,305,624))]:
        s2 = SourceScenario(burst_lat_deg=0.0, burst_lon_deg=0.0, burst_alt_km=alt)
        r = compute_burst_region(s2, model)
        print(f"  {alt:.0f} km: R_up={r['R_up_km']:6.1f}(ref{ref[0]}) "
              f"R_dn={r['R_down_km']:6.1f}(ref{ref[1]}) "
              f"R_tr={r['R_trans_km']:6.1f}(ref{ref[2]})  Rb={r['R_magnetic_confinement_km']:.0f}")

    r = compute_burst_region(sc, model)
    print(f"  485 km, 40.75N 35.5W: L band {r['L_source_min']:.2f}-{r['L_source_max']:.2f}, L_burst={r['L_burst']:.2f}")

    fx = assemble_flux(1.3e25, 1.1, 1.3, h_atm_km=80.0)
    print(f"\n[Starfish flux]    {fx['phi_total_cm2s']:.3e} e/cm^2/s   (target ~1e9)")
    print(f"                   volume={fx['V_m3']:.3e} m^3, mean_E={fx['mean_E_MeV']:.3f} MeV")
