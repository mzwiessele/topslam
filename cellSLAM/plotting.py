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
# * Neither the name of cellSLAM.plotting nor the names of its
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
import numpy as np
from cellSLAM.pseudo_time.distance_correction import _get_label_pos, _get_colors
from GPy.plotting.gpy_plot.plot_util import find_best_layout_for_subplots

def plot_dist_hist(M, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    d_m = M[:,None]-M[None, :]
    D = np.einsum('ijq,ijq->ij',d_m,d_m)
    _ = ax.hist(squareform(D), bins=200)


def plot_comparison(mc, X_init, dims, labels, ulabels, start, cmap='magma', 
                    cmap_index=None, box=True, text_kwargs=None, 
                    adjust=True, adjust_kwargs=dict(arrowprops=dict(arrowstyle="fancy",
                                                                    fc=".6", ec="none")),
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
    ax.text(0.01,.98,"cellSLAM",va='top',transform=ax.transAxes,color='w')

    pt = mc.get_pseudo_time(start=start)
    import itertools
    
    i = 0
    texts = []
    for name in dims:
        ax = next(axit)
        X = X_init[:,dims[name]]
        label_pos, col, mi, ma = _get_label_pos(X, pt, labels, ulabels)
        colors = _get_colors(cmap, col, mi, ma, cmap_index)
        marker = itertools.cycle('<>sd^')
        for l in ulabels:
            #c = Tango.nextMedium()
            c, r = colors[l]
            p = label_pos[l]
            fil = (labels==l)
            ax.scatter(*X[fil].T, linewidth=.1, facecolor=c, alpha=.8, edgecolor='w', marker=next(marker), label=l, **scatter_kwargs)
            rgbc = c#[_c/255. for _c in Tango.hex2rgb(c)]
            if r <.5:
                ec = 'w'
            else:
                ec = 'k'
            if box:
                fc = list(rgbc)
                #fc[-1] = .7
                props = dict(boxstyle='round', facecolor=fc, alpha=0.6, edgecolor=ec)
            else:
                props = dict()
            texts.append(ax.text(p[0], p[1], l, alpha=.9, ha='center', va='center', color=ec, bbox=props, **text_kwargs or {}))
        ax.text(0.01,.98,name,va='top',transform=ax.transAxes)
        i += 2

    from adjustText import adjust_text
    adjust_text(texts, **adjust_kwargs)

    try:
        fig.tight_layout(pad=0)
    except:
        print("Plot Warning: Tight layout failed, continueing without")
    return fig, axes
