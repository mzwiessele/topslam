#===============================================================================
# Copyright (c) 2016, Max Zwiessele
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of topslam.plotting nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================

from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
import numpy as np, itertools
from .pseudo_time.distance_correction import _get_label_pos, _get_colors
from GPy.plotting.gpy_plot.plot_util import find_best_layout_for_subplots

def plot_dist_hist(M, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    d_m = M[:,None]-M[None, :]
    D = np.einsum('ijq,ijq->ij',d_m,d_m)
    _ = ax.hist(squareform(D), bins=200)

def plot_labels_other(X, pt, labels, ulabels, ax=None, cmap='magma', 
                      box=True,
                      adjust=True, 
                         adjust_kwargs=dict(arrowprops=dict(arrowstyle="fancy",
                                                            fc=".6", ec="none",
                                                            ),
                                            ha='center', va='center', force_text=.5, precision=.5),**text_kwargs):
    """
    Plot the labels ontop of the figure.
    
    :param array_like X: two dimensional array for cell positions in landscape.
    :param array_like pt: one dimensional array for pseudotime assignments.
    :param array_like labels: the labels for the cells.
    :param array_like ulabels: the unique labels, for ordering of labels.
    :param axis ax: the matplotlib axis to plot on.
    :param str cmap: the colormap to use for the cells.
    :param bool box: whether to plot a box around the label.
    :param bool adjust: whether to move labels around to not overlap.
    :param dict adjust_kwargs: the keyword arguments to pass to adjust_text
    :param dict text_kwargs: keyword arguments to pass on to ax.text
    """
    if ax is None:
        _, ax = plt.subplots()
    label_pos, col, mi, ma = _get_label_pos(X, pt, labels, ulabels)
    colors = _get_colors(cmap, col, mi, ma, None)
    texts = []
    for l in ulabels:
        #c = Tango.nextMedium()
        c, r = colors[l]
        p = label_pos[l]
        rgbc = c#[_c/255. for _c in Tango.hex2rgb(c)]
        if r <.5:
            ec = 'w'
        else:
            ec = 'k'
        if box:
            fc = list(rgbc)
            #fc[-1] = .7
            props = dict(boxstyle='round', facecolor=fc, alpha=0.6, edgecolor=ec, pad=0.2)
        else:
            props = dict()
        texts.append(ax.text(p[0], p[1], l, alpha=.9, ha='center', va='center', color=ec, bbox=props, **text_kwargs or {}))
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x1, y1 = np.mgrid[xlim[0]:xlim[1]:100j,ylim[0]:ylim[1]:2j]
    x2, y2 = np.mgrid[xlim[0]:xlim[1]:2j,ylim[0]:ylim[1]:100j]
    x, y = np.r_[x1[:,0], x2[1], x1[::-1,1], x2[0]], np.r_[y1[:,0], y2[1], y1[:,1], y2[0,::-1]]
    if adjust:
        from adjustText import adjust_text
        adjust_text(texts, x, y, ax=ax, **adjust_kwargs)
    return ax

def plot_landscape_other(X, pt, labels=None, ulabels=None, ax=None, cmap='magma', cmap_index=None,  
                         coloring='labels', **scatter_kwargs):
    """
    Plot the scatter plot for a landscape.
    
    :param array_like X: two dimensional array for cell positions in landscape.
    :param array_like pt: one dimensional array for pseudotime assignments.
    :param array_like labels: the labels for the cells.
    :param array_like ulabels: the unique labels, for ordering of labels.
    :param axis ax: the matplotlib axis to plot on.
    :param str cmap: the colormap to use for the cells.
    :param int cmap_index: if we choose no labels, which index on the colormap to use for the one color.
    :param str coloring: one of {labels, time}. Indicates what colors to use for the cells, either by label, or by time. 
    """
    if ax is None:
        _, ax = plt.subplots()
    _, col, mi, ma = _get_label_pos(X, pt, labels, ulabels)
    
    marker = itertools.cycle('<>sd^')
    
    if labels is None:
        labels = np.zeros(X.shape[0])
        legend = False
        ulabels = np.zeros(1)
    elif ulabels is None:
        ulabels = []
        for l in labels: 
            if l not in ulabels: ulabels.append(l)
        ulabels = np.asarray(ulabels)
        legend = True
    else:
        legend = True

    if coloring in 'labels':
        colors = _get_colors(cmap, col, mi, ma, cmap_index)
    elif coloring in 'time':
        _cm = plt.get_cmap(cmap)
        _cm = _cm((pt-mi)/(ma-mi))
        
    for l in ulabels:
        fil = (labels==l)
        #c = Tango.nextMedium()
        if coloring in 'labels':
            c, _ = colors[l]
            ax.scatter(*X[fil].T, linewidth=.1, facecolor=c, alpha=.8, edgecolor='w', 
                       marker=next(marker), label=l if legend else None, **scatter_kwargs)
        elif coloring in 'time':
            ax.scatter(*X[fil].T, linewidth=.1, facecolor=_cm[fil], alpha=.8, edgecolor='w', 
                       marker=next(marker), label=l if legend else None, **scatter_kwargs)
    return ax

def plot_comparison(mc, X_init, dims, labels, ulabels, start, cmap='magma', 
                    cmap_index=None, box=True, text_kwargs=None, 
                    adjust=True, adjust_kwargs=dict(arrowprops=dict(arrowstyle="fancy",
                                                                    fc=".6", ec="none",
                                                                    ),
                                                    ha='center', va='center', force_text=.5, precision=.5),
                    **scatter_kwargs):
    fig = plt.figure(figsize=(10,5))
    
    rows, cols = find_best_layout_for_subplots(len(dims)+1)
    gs = plt.GridSpec(rows,cols)
    axes = np.empty((rows,cols), object)

    for r in range(rows):
        for c in range(cols):
            axes[r,c] = fig.add_subplot(gs[(r):((r+1)), c])#, sharex=axes[r-1,c], sharey=axes[r,c-1])
            plt.setp(axes[r,c].get_xticklabels(), visible=False)
            plt.setp(axes[r,c].get_yticklabels(), visible=False)
            plt.setp(axes[r,c].get_xticklines(), visible=False)
            plt.setp(axes[r,c].get_yticklines(), visible=False)
            #axes[r,c].set_axis_bgcolor('none')

    axit = axes.flat
    ax = next(axit)
    #try:
    #    msi = m.get_most_significant_input_dimensions()[:2]
    #except:
    #    msi = m.Y0.get_most_significant_input_dimensions()[:2]
    #m.plot_magnification(labels=labels,
    #                     resolution=20,
    #                     scatter_kwargs=dict(s=40, lw=.1, edgecolor='w'),
    #                     ax=ax,
    ##                     plot_scatter=False,
    #                    legend=False)
    #X = m.X.mean[:, m.get_most_significant_input_dimensions()[:2]]
    #
    #plot_graph_nodes(X, pt, labels, ulabels, ax, cmap=cmap, cmap_index=cmap_index, box=True)
    mc.plot_waddington_landscape(ax=ax)
    mc.plot_graph_nodes(ax=ax, labels=labels, ulabels=ulabels, start=start)
    mc.plot_graph_labels(labels, ulabels=ulabels, ax=ax, start=start)
    
    #for lab in ulabels:
    #    x, y = X[lab==labels].T
    #    ax.scatter(x, y, c=colors[lab], marker=marker[lab], label=lab, lw=0.1, edgecolor='w', s=40)

    #ax.set_xlabel('')
    #ax.set_ylabel('')
    ax.text(0.01,.98,"topslam",va='top',transform=ax.transAxes,color='w')

    pt = mc.get_pseudo_time(start=start)
    
    i = 0
    for name in dims:
        ax = next(axit)
        X = X_init[:,dims[name]]
        plot_landscape_other(X, pt, labels, ulabels, ax, cmap, 
                             cmap_index, box, text_kwargs, 
                             adjust, adjust_kwargs, **scatter_kwargs)
        plot_landscape_other(X, pt, labels, ulabels, ax, cmap, 
                             cmap_index, box, text_kwargs, 
                             adjust, adjust_kwargs, **scatter_kwargs)
        i += 2


    try:
        fig.tight_layout(pad=0)
    except:
        print("Plot Warning: Tight layout failed, continueing without")
    return fig, axes
