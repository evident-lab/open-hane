#!/usr/bin/env python3
"""
Electron dose response behind an aluminium shield.

Dose per unit fluence is S_Si(E) x 1.602e-8 rad per (MeV cm^2/g) per (e/cm^2),
using NIST ESTAR stopping powers and CSDA ranges. Shield transmission is a
range-based approximation and bremsstrahlung is a small parameterised term; a
transport code (e.g. SHIELDOSE-2) would be needed for accurate values.
"""
from __future__ import annotations
import numpy as np

RHO_AL_G_CM3 = 2.70
MEV_TO_RAD_FACTOR = 1.602176634e-8     # rad per (MeV/g) per (particle/cm^2)

# ------------------------------ NIST ESTAR tables ------------------------
def electron_range_al_g_cm2(E_MeV):
    """CSDA range of electrons in Al (g/cm^2), NIST ESTAR."""
    tabE = np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,3.0,5.0,8.0,10.0])
    tabR = np.array([2.7e-4,8.6e-4,4.4e-3,1.86e-2,5.9e-2,1.973e-1,5.546e-1,1.265,1.95,2.806,4.74,5.857])
    E = np.clip(np.asarray(E_MeV,float), tabE[0], tabE[-1])
    return np.exp(np.interp(np.log(E), np.log(tabE), np.log(tabR)))

def electron_mass_stopping_power_si_MeV_cm2_g(E_MeV):
    """Electron collision stopping power in Si (MeV cm^2/g), NIST ESTAR."""
    tabE = np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,8.0,10.0])
    tabS = np.array([20.4,12.8,6.75,4.20,2.78,1.96,1.69,1.67,1.79,1.87,1.91])
    E = np.clip(np.asarray(E_MeV,float), tabE[0], tabE[-1])
    return np.exp(np.interp(np.log(E), np.log(tabE), np.log(tabS)))

# ------------------------------ dose response ----------------------------
def electron_dose_response_rad_per_e_cm2(E_MeV, shield_mm_al, include_brem=True):
    """rad(Si) per incident electron/cm^2 behind an Al-equivalent shield of shield_mm_al.

    Direct term S_Si(E) x 1.602e-8 times a range-based transmission through the
    shield, plus an optional small bremsstrahlung term.
    """
    E = np.asarray(E_MeV, float)
    shield_g_cm2 = shield_mm_al*0.1*RHO_AL_G_CM3
    R = electron_range_al_g_cm2(E)
    # range-based transmission, with a sharper cut-off once the shield exceeds the range
    trans = np.exp(-np.maximum(0.0, shield_g_cm2/np.maximum(R, 1e-9)))
    trans *= np.where(shield_g_cm2 < R, 1.0,
                      np.exp(-(shield_g_cm2-R)/np.maximum(0.05, 0.25*R)))
    S = electron_mass_stopping_power_si_MeV_cm2_g(E)
    direct = MEV_TO_RAD_FACTOR*S*trans
    if not include_brem:
        return direct
    # bremsstrahlung: small, smooth parameterisation
    brem_yield = np.clip(0.0117*E, 0.0, 0.25)
    brem = MEV_TO_RAD_FACTOR*brem_yield*0.15*np.exp(-0.04*shield_g_cm2)
    return direct + brem

# ------------------------------ checks ----------------------------------
if __name__ == "__main__":
    print("Electron dose response checks")

    print("\n[1] Al CSDA range vs NIST ESTAR:")
    for e,ref in {0.1:0.0186,0.5:0.1973,1.0:0.5546,2.0:1.265,5.0:2.806,10.0:5.857}.items():
        m=float(electron_range_al_g_cm2(np.array([e]))[0])
        print(f"   {e:5.2f} MeV: model={m:8.4g} ESTAR={ref:8.4g} ratio={m/ref:.2f}")

    print("\n[2] Si stopping power vs NIST ESTAR:")
    for e,ref in {0.1:4.20,0.5:1.96,1.0:1.69,2.0:1.67,5.0:1.79,10.0:1.91}.items():
        m=float(electron_mass_stopping_power_si_MeV_cm2_g(np.array([e]))[0])
        print(f"   {e:5.2f} MeV: model={m:8.4g} ESTAR={ref:8.4g} ratio={m/ref:.2f}")

    print("\n[3] dose relation (dimensional) + shield cutoff behavior:")
    E=np.array([0.1,0.5,1.0,1.358,2.0,5.0])
    for mm in [0.5, 2.0, 5.0]:
        resp=electron_dose_response_rad_per_e_cm2(E,mm)
        sg=mm*0.1*2.70
        print(f"   shield={mm}mm ({sg:.2f} g/cm2): rad/(e/cm2) = "
              + ", ".join(f"{r:.2e}" for r in resp))
    # physical check: heavier shield -> lower response at low E (stopped)
    r_thin=electron_dose_response_rad_per_e_cm2(np.array([0.5]),0.5)[0]
    r_thick=electron_dose_response_rad_per_e_cm2(np.array([0.5]),5.0)[0]
    print(f"   0.5 MeV: thin(0.5mm)={r_thin:.2e} > thick(5mm)={r_thick:.2e}  "
          f"{'OK (shield stops low-E)' if r_thin>r_thick else 'CHECK'}")

