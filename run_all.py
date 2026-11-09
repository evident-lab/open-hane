"""Rebuild the paper figures and tables.

    python run_all.py           render from analysis/cache (computing anything missing)
    python run_all.py --full    recompute the L grid, the Fig 3 map and the Fig 4 fleet run first
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(ROOT, "analysis")
FIGURES = os.path.join(ROOT, "figures")
CACHE = os.path.join(ANALYSIS, "cache")
PY = sys.executable


def run(*args, cwd=ANALYSIS):
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=cwd, check=True)


def missing(name):
    return not os.path.exists(os.path.join(CACHE, name))


if __name__ == "__main__":
    full = "--full" in sys.argv
    if full or missing("Lgrid_2026014.npz"):
        run("lgrid.py")
    if full or missing("fig3_peakmap.npz"):
        procs = [subprocess.Popen([PY, "fig3_compute.py", str(p), "2"], cwd=ANALYSIS) for p in (0, 1)]
        if any(p.wait() for p in procs):
            sys.exit("fig3_compute failed")
        run("-c", "import fig3_compute as f; f.merge(2)")
    if full or missing("fleet_saa.npz"):
        run("fleet_compute.py", "-20", "-35", "500", "saa")

    for fig in ("fig2_belt_cross_sections", "fig3_burst_location_map", "fig4_fleet_loss"):
        run(os.path.join(FIGURES, fig, "render.py"), cwd=os.path.join(FIGURES, fig))
    run("tables.py")
    print("done: figures in figures/*, tables in analysis/tables")
