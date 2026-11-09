#!/usr/bin/env python3
"""
Prompt X-ray fluence and dose at a satellite.

Blackbody spectrum (kT = 1 keV by default), attenuated along the line of sight by
the air column and then by an aluminium shield; dose in silicon from the mass
energy-absorption coefficient. Lines of sight through the Earth give zero.
Air attenuation is a piecewise fit; Al and Si coefficients are NIST XCOM values.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

R_E_KM = 6371.0
J_PER_KT = 4.184e12
RHO_AL_G_CM3 = 2.70

def builtin_density_kg_m3(h_km):
    z = np.array([0,10,20,30,40,50,60,70,80,90,100,120,150,180,200,250,300,400,500,600,700,800,900,1000],float)
    rho = np.array([1.225,0.4135,0.08891,0.01841,0.003996,1.027e-3,3.097e-4,8.283e-5,1.846e-5,
                    3.416e-6,5.6e-7,2.4e-8,2.0e-9,5.0e-10,2.5e-10,6.0e-11,2.0e-11,3.0e-12,
                    1.0e-12,5.0e-13,2.5e-13,1.2e-13,6.0e-14,3.0e-14],float)
    hk = np.clip(np.asarray(h_km,float), z[0], z[-1])
    return np.exp(np.interp(hk, z, np.log(rho)))

def column_mass_slant_g_cm2(r0_km, r1_km, ds_km=5.0, atm_cut_km=1000.0):
    d = r1_km - r0_km; Ln = float(np.linalg.norm(d))
    if Ln <= 0: return 0.0
    n = min(max(2, int(math.ceil(Ln/ds_km))+1), 5000)
    s = np.linspace(0.0, Ln, n)
    r = r0_km[None,:] + (s[:,None]/Ln)*d[None,:]
    h = np.linalg.norm(r,axis=1) - R_E_KM
    keep = h <= atm_cut_km
    if not np.any(keep): return 0.0
    last = np.where(keep)[0][-1]
    s = s[:last+1]; h = np.maximum(h[:last+1],0.0)
    sigma_kg_m2 = trapz(builtin_density_kg_m3(h), s*1000.0)
    return float(sigma_kg_m2/10.0)

def mu_air_cm2_g(E_keV):
    E = np.maximum(np.asarray(E_keV,float),0.01); mu = np.empty_like(E)
    m1=E<=5.0; mu[m1]=(3325.18643*np.exp(-1.0507*E[m1])+61686.30854*np.exp(-3.2391*E[m1])+25.0528)
    m2=(E>5.0)&(E<=30.0); mu[m2]=(20.44783*np.exp(-0.1779*E[m2])+607.57386*np.exp(-0.6041*E[m2])+0.1801)
    m3=(E>30.0)&(E<=40.0); mu[m3]=-0.01053*E[m3]+0.6697
    m4=(E>40.0)&(E<=50.0); mu[m4]=-0.00405*E[m4]+0.4105
    m5=E>50.0; mu[m5]=np.maximum(0.02,0.25*(E[m5]/50.0)**-0.35)
    return np.maximum(mu,1e-6)

def mu_al_cm2_g(E_keV):
    # NIST XCOM total mass attenuation coefficient of Al (cm^2/g), log-log interpolation
    tab_E=np.array([0.1,0.2,0.3,0.5,1.0,1.5,2.0,3.0,5.0,10.0,20.0,30.0,50.0,100.0,200.0,500.0,1000.0,5000.0,10000.0])
    tab_mu=np.array([41600.0,26200.0,12100.0,5430.0,1185.0,402.0,226.0,78.8,19.3,26.23,3.39,1.128,0.3681,0.1704,0.1223,0.0844,0.06146,0.02793,0.02318])
    return np.exp(np.interp(np.log(np.clip(E_keV,tab_E[0],tab_E[-1])),np.log(tab_E),np.log(tab_mu)))

def mu_en_si_cm2_g(E_keV):
    # NIST mass energy-absorption coefficient of Si (cm^2/g), log-log interpolation
    tab_E=np.array([0.1,0.2,0.3,0.5,1.0,1.5,2.0,3.0,5.0,10.0,20.0,30.0,50.0,100.0,200.0,500.0,1000.0,5000.0,10000.0])
    tab_mu=np.array([1300.0,2050.0,2950.0,3190.0,1570.0,680.0,389.0,139.0,35.9,33.6,4.93,1.61,0.392,0.0386,0.0238,0.0245,0.0277,0.0242,0.0214])
    return np.exp(np.interp(np.log(np.clip(E_keV,tab_E[0],tab_E[-1])),np.log(tab_E),np.log(tab_mu)))

def blackbody_energy_pdf(E_keV, kT_keV):
    E=np.asarray(E_keV,float); x=np.clip(E/kT_keV,1e-12,700.0)
    b=(15.0/math.pi**4)*(E**3)/(kT_keV**4)/np.expm1(x)
    return b/max(trapz(b,E),1e-300)

def is_occulted_segment(r0_km, r1_km, R_block_km=R_E_KM):
    d=r1_km-r0_km; dd=float(np.dot(d,d))
    if dd<=0: return False
    t=max(0.0,min(1.0,-float(np.dot(r0_km,d))/dd))
    return float(np.linalg.norm(r0_km+t*d))<R_block_km

@dataclass
class PromptScenario:
    yield_kt: float = 1000.0
    xray_fraction: float = 0.70
    kT_keV: float = 1.0
    prompt_E_min_keV: float = 0.1
    # The spectrum runs to 50 keV: a 1 keV blackbody has about 40% of its energy
    # above 4 keV. The vacuum fluence does not depend on this cutoff (the spectral
    # factor is normalised); only heavily attenuated paths change, by a few per cent.
    prompt_E_max_keV: float = 50.0
    prompt_nE: int = 300
    prompt_ds_km: float = 5.0
    prompt_atm_cut_km: float = 1000.0
    shield_mm_al: float = 2.0

def lla_to_ecef_km(lat,lon,alt):
    la,lo=math.radians(lat),math.radians(lon); r=R_E_KM+alt
    return np.array([r*math.cos(la)*math.cos(lo),r*math.cos(la)*math.sin(lo),r*math.sin(la)])

def prompt_xray_dose_krad_si(r_burst_km, r_sat_km, sc: PromptScenario) -> Tuple[float,float,float]:
    if is_occulted_segment(r_burst_km, r_sat_km): return 0.0,0.0,0.0
    R_m = float(np.linalg.norm(r_sat_km-r_burst_km))*1000.0
    if R_m<=0: return 0.0,0.0,0.0
    E_x_J = sc.yield_kt*J_PER_KT*sc.xray_fraction
    F_vac = E_x_J/(4.0*math.pi*R_m*R_m)
    sigma_air = column_mass_slant_g_cm2(r_burst_km,r_sat_km,sc.prompt_ds_km,sc.prompt_atm_cut_km)
    E = np.geomspace(sc.prompt_E_min_keV,sc.prompt_E_max_keV,sc.prompt_nE)
    pdf = blackbody_energy_pdf(E,sc.kT_keV)
    shield_g_cm2 = sc.shield_mm_al*0.1*RHO_AL_G_CM3
    T_air=np.exp(-mu_air_cm2_g(E)*sigma_air); T_al=np.exp(-mu_al_cm2_g(E)*shield_g_cm2)
    muen=mu_en_si_cm2_g(E)*0.1
    resp=trapz(pdf*T_air*T_al*muen,E)
    dose_krad=(F_vac*resp)/10.0
    F_after_air=F_vac*trapz(pdf*T_air,E)
    return float(dose_krad),float(F_after_air),float(sigma_air)

# ------------------------------ checks ----------------------------------
if __name__ == "__main__":
    print("Prompt X-ray checks")
    sc = PromptScenario()

    # (1) vacuum fluence inverse-square (no atmosphere, no shield, space-space)
    print("\n[1] vacuum fluence inverse-square:")
    r_b = lla_to_ecef_km(0,0,485)
    for Rkm in [100.0, 1000.0, 5000.0]:
        r_s = r_b + np.array([0.0,0.0,Rkm])      # straight up, no Earth in path
        _,F,sig = prompt_xray_dose_krad_si(r_b, r_s, sc)
        E_x=sc.yield_kt*J_PER_KT*sc.xray_fraction
        F_exact=E_x/(4*math.pi*(Rkm*1000)**2)
        print(f"   R={Rkm:6.0f} km: F_after_air={F:.4e}  F_vac_exact={F_exact:.4e}  sigma_air={sig:.3e} g/cm2")

    # (2) blackbody pdf
    print("\n[2] blackbody spectrum:")
    E=np.geomspace(0.1,50,300); pdf=blackbody_energy_pdf(E,1.0)
    print(f"   integral={trapz(pdf,E):.4f} (expect 1.0), peak at E={E[np.argmax(pdf)]:.2f} keV (Wien~2.82 keV)")

    # (3) attenuation coefficients against tabulated NIST values
    print("\n[3] attenuation vs NIST (cm^2/g):")
    print("    Al: mu/rho NIST ~ 25.7@1keV, 3.4@10keV(*total w/coh), ~0.07@1MeV")
    for e in [1.0,10.0,1000.0]:
        print(f"      Al({e:6.1f} keV) model={float(mu_al_cm2_g(np.array([e]))[0]):.3g}")
    print("    Si muen NIST ~ 1570@1keV, ~0.028@1MeV")
    for e in [1.0,1000.0]:
        print(f"      Si_muen({e:6.1f} keV) model={float(mu_en_si_cm2_g(np.array([e]))[0]):.3g}")

    # (4) occultation -> zero
    print("\n[4] occultation:")
    r_far = lla_to_ecef_km(0,180,485)   # opposite side of Earth
    d,_,_ = prompt_xray_dose_krad_si(r_b, r_far, sc)
    print(f"   dose through Earth = {d:.3e} krad (expect 0)")

    # (5) sample LEO dose magnitude
    print("\n[5] sample dose, sat 1200 km slant, 2mm Al:")
    r_s = lla_to_ecef_km(10,5,800)
    dose,F,sig = prompt_xray_dose_krad_si(r_b,r_s,sc)
    print(f"   dose={dose:.4g} krad(Si), fluence_after_air={F:.4g} J/m2, sigma_air={sig:.3g} g/cm2")