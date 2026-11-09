import matplotlib as mpl

def plotter_style():

    mpl.rcParams.update({
        # figure
        "figure.figsize": (6.5, 4.5),
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # font / typography
        "font.family": "Arial",
        "font.size": 10,
        "mathtext.fontset": "stix",
        # axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#222222",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.formatter.use_mathtext": True,
        "axes.formatter.limits": (-3, 4),
        "axes.formatter.useoffset": False,
        # color cycle
        "axes.prop_cycle": mpl.cycler(
            color=[
                "#163A70",   # navy
                "#555555",   # dark grey
                "#8A8A8A",   # grey
                "#8B1E1E",   # dark red
                "#2C6E49",   # muted green
                "#B07C22",   # muted gold
                "#5C4B7D",   # muted violet
            ]
        ),
        # lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.0,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        # x ticks
        "xtick.color": "#222222",
        "xtick.labelsize": 9,
        "xtick.direction": "out",
        "xtick.major.size": 5,
        "xtick.major.width": 1.0,
        "xtick.minor.visible": True,
        "xtick.minor.size": 3,
        "xtick.minor.width": 0.8,
        # y ticks
        "ytick.color": "#222222",
        "ytick.labelsize": 9,
        "ytick.direction": "out",
        "ytick.major.size": 5,
        "ytick.major.width": 1.0,
        "ytick.minor.visible": True,
        "ytick.minor.size": 3,
        "ytick.minor.width": 0.8,
        # grid
        "grid.color": "#E5E5E5",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        # legend
        "legend.frameon": False,
        "legend.fancybox": False,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 9,
        "legend.handlelength": 2.8,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.4,
        "legend.borderaxespad": 0.5,
        # patches
        "patch.linewidth": 1.0,
        "patch.edgecolor": "#222222",
        # error bars
        "errorbar.capsize": 3,
        # image / contour maps
        "image.cmap": "cividis",
        # pdf / svg export
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
