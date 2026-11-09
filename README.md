# open-HANE

Code for the radiation-belt part of the paper: beta trapping in the IGRF-14 field, evolution of the
artificial electron belt, and prompt X-ray plus belt dose to the LEO fleet (CelesTrak, 2026-01-14).
It produces Figs 2-4 and the supplementary tables of radiation-driven fleet loss.

## Layout

```
hanetrap/       IGRF field, field-line tracing, drift-shell loss table, debris expansion region, trapping S
  data/         IGRF-14 coefficients, US Standard Atmosphere 1976, U-235 beta-activity tables
  tests/        pytest checks
openhane_src/   fission-electron source, SGP4 orbits, prompt X-rays, radial diffusion, dose response
analysis/       calculations behind the figures and tables
  cache/        saved runs so the figures render without recomputing
  tables/       CSV and LaTeX versions of the supplementary tables, and the Fig 4 curves
figures/        one folder per figure: render.py and the PNG/PDF it writes
data/           CelesTrak TLEs (2026-01-14 02:00 CT) and Natural Earth coastlines
run_all.py      renders every figure and table
```

## Setup

Python 3.11 or newer.

```
pip install -r requirements.txt
```

## Reproducing the paper

```
python run_all.py          # figures and tables from analysis/cache
python run_all.py --full   # also recompute the L grid, the Fig 3 map and the Fig 4 run
```

The first run builds the IGRF drift-shell table in `hanetrap/outputs/` (about 5 minutes); later runs reuse it.
`--full` takes several hours, mostly the Fig 3 scan over burst locations.

| Paper item | Calculation | Plot |
|---|---|---|
| Fig 2, belt cross-sections | inside the render script | `figures/fig2_belt_cross_sections/render.py` |
| Fig 3, peak dose rate vs burst location | `analysis/fig3_compute.py` | `figures/fig3_burst_location_map/render.py` |
| Fig 4, fleet loss vs time | `analysis/fleet_compute.py -20 -35 500 saa` | `figures/fig4_fleet_loss/render.py` |
| Supp. table, loss by source case and tolerance | `analysis/tables.py` | |
| Supp. table, loss by yield | one-year runs below, then `analysis/tables.py` | |

The collision and debris results (Fig 5, the collision supplementary figure and tables) and the
satellite altitude histogram come from separate code and are not included here.

## Yield runs

`fleet_compute.py LAT LON ALT TAG` writes `analysis/cache/fleet_<TAG>.npz`. `HANE_YIELD_KT` sets the
yield in kt (default 1000) and `HANE_DAYS` the length of the run (default 60). The yield table uses
three one-year runs:

```
cd analysis
HANE_DAYS=365 python fleet_compute.py -20 -35 500 saa_1mt_365d
HANE_DAYS=365 HANE_YIELD_KT=100 python fleet_compute.py -20 -35 500 saa_100kt
HANE_DAYS=365 HANE_YIELD_KT=10  python fleet_compute.py -20 -35 500 saa_10kt
python tables.py
```

Each takes about an hour. The outputs are 50-100 MB, so they are not kept in the repository;
`analysis/tables/yield_loss.csv` has the results.

## Model summary

- Yield: 1 Mt total, 50% fission (7.25e25 fissions, 1.45e23 per kt; Glasstone & Dolan 1977).
- Field: IGRF-14 at 2026-01-14, loss altitude 120 km, L = (31,200 nT / B_min)^(1/3).
- Trapping: S = sqrt(1 - B/B_c) averaged over each cell's drift shell; 0.404, 1.289 and 5.207
  electrons per fission (central case 1.289).
- Belt: Carter spectrum, Brautigam-Albert D_LL at Kp = 2, empirical lifetimes, dipole drift-shell
  volumes above 80 km.
- Dose: 2 mm Al, NIST ESTAR stopping powers and ranges.
- Fleet: SGP4 every 20 minutes; a satellite fails when the prompt X-ray fluence reaches 40 J/m^2 or
  the total dose reaches 40 or 120 krad(Si).

## Tests

```
cd hanetrap && python -m pytest tests
```
