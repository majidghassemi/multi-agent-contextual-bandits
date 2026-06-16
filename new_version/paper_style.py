"""
paper_style.py  --  publication-quality matplotlib styling for the CEC paper.

Usage
-----
    import paper_style as ps
    ps.set_style()                      # apply globally, once, before plotting
    fig, ax = ps.figure(width='single') # 'single' (3.25in) or 'double' (6.75in) column
    ...
    ps.finalize(ax, xlabel=..., ylabel=..., title=...)
    ps.save(fig, 'fig_exp2')            # writes BOTH .pdf (vector, for LaTeX) and .png

Design choices (aligned with NeurIPS/ICML/AISTATS house style):
  * Serif font matching LaTeX body text; mathtext in the same family.
  * Single-column width 3.25in so the figure is legible at print size with no
    further scaling in LaTeX (\\includegraphics[width=\\columnwidth]).
  * A restrained, colour-blind-safe palette (Wong 2011) instead of matplotlib
    defaults; line + marker + (optional) linestyle so curves are distinguishable
    in greyscale print.
  * Light, thin gridlines; spines trimmed; ticks pointing out; no chartjunk.
  * Vector PDF output is the deliverable for the camera-ready; PNG is a preview.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Wong (2011) colour-blind-safe palette -------------------------------------
PALETTE = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'sky':    '#56B4E9',
    'yellow': '#F0E442',
    'grey':   '#999999',
    'black':  '#000000',
}
CYCLE = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'],
         PALETTE['red'], PALETTE['purple'], PALETTE['sky']]

# inches; matches a typical two-column ML paper
WIDTHS = {'single': 3.25, 'double': 6.75}


def set_style():
    """Apply the global rcParams.  Call once before creating any figure."""
    mpl.rcParams.update({
        # fonts -- serif to match LaTeX body; mathtext in the same family
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        # lines / markers
        'lines.linewidth': 1.6,
        'lines.markersize': 4.5,
        'axes.prop_cycle': mpl.cycler(color=CYCLE),
        # spines & ticks -- trim top/right, ticks pointing out
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.linewidth': 0.8,
        # grid -- light, thin, behind data
        'axes.grid': True,
        'grid.color': '#CCCCCC',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
        'axes.axisbelow': True,
        # legend -- light frame, no shadow
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': False,
        'legend.borderpad': 0.4,
        # figure / saving
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,    # editable/embeddable TrueType (camera-ready safe)
        'ps.fonttype': 42,
    })


def figure(width='single', height_ratio=0.72, nrows=1, ncols=1):
    """Create a correctly-sized figure.  height_ratio is height/width per panel."""
    w = WIDTHS[width]
    h = w * height_ratio
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    return fig, ax


def finalize(ax, xlabel=None, ylabel=None, title=None, legend=True, legend_loc='best'):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=4)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc)


def save(fig, stem, outdir='.'):
    """Write both a vector PDF (for LaTeX) and a PNG preview."""
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outdir}/{stem}.{ext}')
    plt.close(fig)