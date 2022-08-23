import matplotlib
import numpy as np
from cycler import cycler


def set_figure(fontsize=9, width=251.0, heightratio=None, height=None):
    r"""
    Parameters
    ----------
    fontsize : float
        sets the intended fontsize

    width : float
        sets the intended width in pts

    Notes
    -----
    To set equal to the columnwidth of the article:

    In the tex file 'Column width: \the\columnwidth' will print this size
    alternatively, '\message{Column width: \the\columnwidth}' will print to the log

    \linewidth should be used in place of \columnwidth if the figure is used
    within special enviroments (e.g. minipage)

    https://matplotlib.org/stable/tutorials/introductory/customizing.html
    https://scipy-cookbook.readthedocs.io/items/Matplotlib_LaTeX_Examples.html
    https://tex.stackexchange.com/questions/16942/difference-between-textwidth-linewidth-and-hsize
    """
    fig_width_pt = width
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches

    if heightratio is None:
        heightratio = (np.sqrt(5)-1.0)/2.0  # Aesthetic ratio
    if height is None:
        fig_height = fig_width*heightratio      # height in inches
    else:
        fig_height = height*inches_per_pt
    fig_size = [fig_width, fig_height]
    params = {'backend': 'pdf',
              'text.usetex': True,
              'text.latex.preamble': r"""
                                      \usepackage{lmodern}
                                      \usepackage[cm]{sfmath}
                                      \usepackage[T1]{fontenc}
                                      """,
              # fonts
              'font.family': 'sans-serif',
              # font sizes
              'axes.labelsize': fontsize,
              'font.size': fontsize,
              'axes.titlesize': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              # figure size
              'figure.figsize': fig_size,
              'figure.constrained_layout.use': True,
              # legend
              'legend.frameon': False,
              # spines
              'axes.spines.top': True,
              'axes.spines.right': True,
              # saving
              'savefig.bbox': 'tight',
              'savefig.pad_inches': 1/72,
              # grid
              'grid.color': '0.7',
              'grid.linewidth': 0.2,
              # ticks
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              }
    matplotlib.rcParams.update(params)
