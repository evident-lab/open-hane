"""Supplementary tables: fleet loss for the electron-source and tolerance cases, and fleet loss by yield.

Inputs: cache/fleet_saa.npz for the first table, and the one-year runs
cache/fleet_saa_1mt_365d.npz, fleet_saa_100kt.npz, fleet_saa_10kt.npz for the yield table
(see README). Writes CSV and LaTeX to analysis/tables/, plus fig4_curves.csv with the Fig 4 curves.
"""
import csv, os
import numpy as np
from fleet_compute import loss_curve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables"); os.makedirs(OUT, exist_ok=True)
CASES = (("low", "0.404"), ("mid", "1.289"), ("high", "5.207"))
KRADS = (40.0, 120.0)


def write(name, header, rows, latex_caption, latex_label, fmt=None):
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    cols = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[H]\centering\small", rf"\caption{{{latex_caption}}}\label{{{latex_label}}}",
             (r"\resizebox{\textwidth}{!}{" if len(header) >= 8 else "") + rf"\begin{{tabular}}{{{cols}}}",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(x) for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if len(header) >= 8 else ""), r"\end{table}"]
    open(os.path.join(OUT, f"{name}.tex"), "w").write("\n".join(lines) + "\n")


def first(t, c, level):
    k = np.nonzero(c >= level)[0]
    return f"{t[k[0]]:.2f}" if len(k) else "--"



if __name__ == "__main__":
    # ---- fleet loss for each source case and tolerance (Supplementary Table: radiation_loss_cases)
    f = np.load(os.path.join(HERE, "cache", "fleet_saa.npz")); t = f["times"] / 86400.0
    rows = []; curves = {}
    for K in KRADS:
        for case, b in CASES:
            c = loss_curve(f, case, K); curves[(K, case)] = c
            rows.append([f"{K:.0f}", b] + [f"{c[np.argmin(abs(t - D))]:.1f}" for D in (1, 3, 7, 15, 30)]
                        + [first(t, c, L) for L in (50, 90)])
    write("radiation_loss_cases", ["Limit (krad)", r"$\beta$", "Day 1 (\\%)", "Day 3", "Day 7", "Day 15", "Day 30",
                                "$t_{50\\%}$ (d)", "$t_{90\\%}$ (d)"], rows,
          r"Fraction of the LEO fleet (13{,}306 objects at 0--2000 km) lost after a 1 Mt burst at 500 km, "
          rf"20$^\circ$S 35$^\circ$W. Prompt X-ray loss at $t=0$: {100*np.mean(f['Fp'] >= 40):.1f}\%.", "tab:fleet")
    with open(os.path.join(OUT, "fig4_curves.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["day"] + [f"loss_{int(K)}krad_{b}beta_pct" for K in KRADS for _, b in CASES])
        for k in range(len(t)):
            w.writerow([f"{t[k]:.3f}"] + [f"{curves[(K, case)][k]:.3f}" for K in KRADS for case, _ in CASES])

    # ---- fleet loss by yield, central source (Supplementary Table: yield_loss)
    rows = []
    for label, tag in (("1 Mt", "saa_1mt_365d"), ("100 kt", "saa_100kt"), ("10 kt", "saa_10kt")):
        path = os.path.join(HERE, "cache", f"fleet_{tag}.npz")
        if not os.path.exists(path):
            print("missing", path, "- yield table row skipped")
            continue
        d = np.load(path); ty = d["times"] / 86400.0
        for K in KRADS:
            c = loss_curve(d, "mid", K)
            rows.append([label, f"{K:.0f}"] + [f"{c[np.argmin(abs(ty - D))]:.1f}" for D in (1, 7, 30, 60, 180, 365)]
                        + [first(ty, c, L) for L in (50, 90)])
    if rows:
        write("yield_loss", ["Yield", "Limit (krad)", "Day 1 (\\%)", "Day 7", "Day 30", "Day 60", "Day 180",
                             "Day 365", "$t_{50\\%}$ (d)", "$t_{90\\%}$ (d)"], rows,
              r"Fleet loss for 1 Mt, 100 kt and 10 kt at 500 km, 20$^\circ$S 35$^\circ$W, 1.29 electrons per fission.",
              "tab:yield_loss")
    print("tables written to", OUT)
