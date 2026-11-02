"""Shared figure typography and colors of the paper."""

# The same blue, teal and terracotta roles as the framework figure.
INK, TEAL, BLUE, ORANGE = "#000000", "#287E78", "#356A8A", "#B45B22"
FONT_SIZE = 10 * 72 / 72.27  # IJCAI body size: TeX points converted to PDF points


def paper_plot_style():
    """10 pt labels at the actual 7 in paper / 3.375 in column widths."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": FONT_SIZE,
            "text.usetex": False,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.bfit": "Times New Roman:italic:bold",
            "mathtext.cal": "Times New Roman:italic",
            "mathtext.sf": "Times New Roman",
            "mathtext.tt": "Times New Roman",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )
