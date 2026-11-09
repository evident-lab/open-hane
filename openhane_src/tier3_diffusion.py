#!/usr/bin/env python3
"""
Radial diffusion and loss of the trapped electron population.

dN/dt = d/dL (D_LL dN/dL) - N / tau + Q, solved by backward Euler with zero-flux
boundaries. D_LL is the Brautigam-Albert (2000) magnetic coefficient. Electron
lifetimes are an empirical parameterisation, not a quasi-linear calculation, so
decay times should be treated as uncertain. The operator acts on N directly rather
than on phase-space density.
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
R_E_KM = 6371.0
R_E_M  = 6371.0e3

# ------------------------------ diffusion --------------------------------
def brautigam_albert_DLL_per_s(L, Kp):
    """Brautigam & Albert (2000) radial diffusion coefficient.
    D_LL = 10^(0.506*Kp - 9.325) * L^10  [1/day], returned in [1/s]."""
    K = float(np.clip(Kp, 1.0, 6.0))
    D_day = (10.0 ** (0.506*K - 9.325)) * (np.asarray(L, float) ** 10)
    return D_day/86400.0

def thomas_solve(lower, diag, upper, rhs):
    n = len(diag); c = np.zeros(n-1); d = np.zeros(n)
    c[0] = upper[0]/diag[0]; d[0] = rhs[0]/diag[0]
    for i in range(1, n-1):
        den = diag[i] - lower[i-1]*c[i-1]
        c[i] = upper[i]/den; d[i] = (rhs[i] - lower[i-1]*d[i-1])/den
    if n > 1:
        den = diag[-1] - lower[-1]*c[-1]
        d[-1] = (rhs[-1] - lower[-1]*d[-2])/den
    x = np.zeros(n); x[-1] = d[-1]
    for i in range(n-2, -1, -1):
        x[i] = d[i] - c[i]*x[i+1]
    return x

def drift_shell_volume_bin_m3(L1, L2, h_atm_km=80.0, nL=8, nlam=80):
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

# ------------------------------ electron lifetimes -----------------------
def loss_tau_s(E_MeV, L_grid):
    """Older order-of-magnitude lifetime (s), kept for comparison.

    It increases monotonically with L, which is the wrong trend for the inner belt;
    the pipeline uses loss_tau_s_empirical.
    """
    E = np.asarray(E_MeV, float)[:, None]; L = np.asarray(L_grid, float)[None, :]
    tau_days = 100.0*(np.maximum(E, 0.05)/1.5)**0.35*np.exp((L-2.0)/0.55)
    tau_days *= np.where(L < 1.25, 0.05, 1.0)
    tau_days = np.clip(tau_days, 0.25, 5.0*365.0)
    return tau_days*86400.0


def loss_tau_s_empirical(E_MeV, L_grid):
    """Empirical electron lifetime (s) on an (energy, L) grid.

    Follows the observed radial structure of inner-zone and slot losses:
    lifetimes of years near L = 1.4, a minimum of about two weeks in the slot
    (L ~ 3), weeks to months beyond it, and about a day below L ~ 1.1 where
    electrons reach the atmosphere. A weak energy dependence (E^0.3) is applied
    and the result is limited to 0.5 days - 8 years.

    This is a parameterisation guided by Starfish observations and published
    inner-belt hiss lifetimes (Abel & Thorne 1998; Ripoll et al. 2016;
    Claudepierre et al. 2020, 2022). It has not been fitted to a specific data set.
    """
    E = np.asarray(E_MeV, float)[:, None]; L = np.asarray(L_grid, float)[None, :]

    TAU_INNER = 1500.0     # days, inner-belt maximum near L = 1.4
    TAU_SLOT = 15.0        # days, slot-region minimum
    TAU_OUTER = 60.0       # days, beyond the slot

    g_inner = np.exp(-((L - 1.4) / 0.55) ** 2)          # inner-belt term, peaks at L = 1.4
    g_outer = 1.0 / (1.0 + np.exp(-(L - 3.5) / 0.4))    # recovery beyond the slot
    tau_days = TAU_SLOT + (TAU_INNER - TAU_SLOT) * g_inner + (TAU_OUTER - TAU_SLOT) * g_outer

    atm = np.clip((L - 1.08) / 0.10, 0.0, 1.0)          # atmospheric cutoff at low L
    tau_days = tau_days * atm + 1.0 * (1.0 - atm)

    tau_days = tau_days * (np.maximum(E, 0.1) / 0.5) ** 0.3
    tau_days = np.clip(tau_days, 0.5, 8.0 * 365.0)
    return tau_days * 86400.0


def loss_tau_s_by_name(name, E_MeV, L_grid):
    """Lifetime by model name: 'empirical' (used in the paper) or anything else for the older form."""
    if str(name).lower() == "empirical":
        return loss_tau_s_empirical(E_MeV, L_grid)
    return loss_tau_s(E_MeV, L_grid)

# ------------------------------ time step --------------------------------
def implicit_diffusion_step(N, L, D_per_s, dt_s, inv_tau=None, source=0.0):
    """One backward-Euler step of radial diffusion with optional loss and source.

    N and L are 1-D, D_per_s is on the L grid, boundaries are zero-flux.
    source is the number added during the step. Returns the updated N.
    """
    nL = len(L); dL = L[1]-L[0]
    D_face = 0.5*(D_per_s[:-1] + D_per_s[1:])
    aL = np.zeros(nL); aR = np.zeros(nL)
    aL[1:] = D_face/(dL*dL); aR[:-1] = D_face/(dL*dL)
    loss = np.zeros(nL) if inv_tau is None else np.asarray(inv_tau, float)
    diag = 1.0 + dt_s*(aL + aR + loss)
    lower = -dt_s*aL[1:]; upper = -dt_s*aR[:-1]
    rhs = N + (source if np.isscalar(source) else np.asarray(source))
    return thomas_solve(lower, diag, upper, rhs)

# ------------------------------ checks ----------------------------------
if __name__ == "__main__":
    print("Diffusion checks")

    # (1) D_LL formula hand-check
    print("\n[1] Brautigam-Albert D_LL:")
    for Kp, L in [(2.0, 4.0), (4.0, 5.0)]:
        got = float(brautigam_albert_DLL_per_s(L, Kp))
        hand = (10.0**(0.506*Kp - 9.325))*(L**10)/86400.0
        print(f"   Kp={Kp}, L={L}: D={got:.4e} /s  hand={hand:.4e} /s  "
              f"{'OK' if abs(got-hand)<1e-30 or abs(got/hand-1)<1e-12 else 'MISMATCH'}")

    # (2) Thomas solver vs numpy
    print("\n[2] Thomas solver vs numpy.linalg.solve:")
    rng = np.random.default_rng(0); n = 50
    lo = rng.uniform(-1,-0.1,n-1); up = rng.uniform(-1,-0.1,n-1)
    di = rng.uniform(3,5,n)  # diagonally dominant
    rhs = rng.uniform(-1,1,n)
    A = np.diag(di) + np.diag(lo,-1) + np.diag(up,1)
    x_thomas = thomas_solve(lo, di, up, rhs); x_numpy = np.linalg.solve(A, rhs)
    err = float(np.max(np.abs(x_thomas-x_numpy)))
    print(f"   max diff = {err:.2e}  {'OK' if err<1e-10 else 'MISMATCH'}")

    # (3) conservation: no source, no loss -> total N conserved
    print("\n[3] diffusion conservation (no source/loss):")
    L = np.linspace(1.1, 6.0, 120)
    N = np.exp(-0.5*((L-2.4)/0.2)**2)   # a localized injected band
    D = brautigam_albert_DLL_per_s(L, 3.0)
    tot0 = np.trapezoid(N, L)
    for _ in range(200):
        N = implicit_diffusion_step(N, L, D, dt_s=6*3600.0)
        N = np.maximum(N, 0.0)
    tot1 = np.trapezoid(N, L)
    print(f"   total before={tot0:.5f}, after 200 steps={tot1:.5f}, "
          f"drift={100*(tot1-tot0)/tot0:+.2f}%")
    print("   (zero-flux boundaries: the total should stay constant)")

    # (4) dipole drift-shell volume
    print(f"\n[4] drift_shell_volume_bin_m3 L=1.1-1.3 (h=80) = "
          f"{drift_shell_volume_bin_m3(1.1,1.3,80.0):.3e} m^3")

    # (5) sample lifetimes
    tau = loss_tau_s_empirical(np.array([1.0]), np.array([1.2, 1.4, 2.0, 3.0]))[0]/86400.0
    print("\n[5] empirical lifetime at 1 MeV, L = 1.2, 1.4, 2.0, 3.0: "
          + ", ".join(f"{x:.0f}" for x in tau) + " days")