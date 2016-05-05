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
import numpy as np, networkx as nx, itertools
from .landscape import waddington_landscape

def plot_dist_hist(M, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    d_m = M[:,None]-M[None, :]
    D = np.einsum('ijq,ijq->ij',d_m,d_m)
    _ = ax.hist(squareform(D), bins=200)

def plot_waddington_landscape_3d(mc, labels=None, ulabels=None, resolution=60, ncol=5, cmap='terrain', cstride=1, rstride=1, xmargin=(.075, .075), ymargin=(.075, .075), **kw):
    """
    Plot a waddngton landscape with data in 3D.
    Xgrid and wad are the landscape (surface plot, [resolution x resolution])
    and X and wadX are the datapoints as returned by
    Xgrid, wadXgrid, X, wadX = landscape(m).

    ulabels and labels are the unique labels and labels for each datapoint of X.
    ncol defines the number of columns in the legend above the plot.

    Returns the 3d axis instance of mplot3d.
    """    
    if labels is None:
        labels = np.zeros(mc.gplvm.X.shape[0])

    if ulabels is None:
        ulabels = []
        for l in labels:
            if l not in ulabels:
                ulabels.append(l)
        ulabels = np.asarray(ulabels)
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(4.66666666655,3.5), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    from GPy.plotting import Tango
    Tango.reset()

    from itertools import cycle
    colors = cycle(Tango.mediumList)
    markers = cycle('<>^vsd')

    r = lambda x: x.reshape(resolution, resolution).T
    
    (Xgrid, wadXgrid, X, wadX) = waddington_landscape(mc.gplvm, resolution, xmargin, ymargin)

    ax.plot_surface(r(Xgrid[:,0]), r(Xgrid[:,1]), r(wadXgrid), cmap=cmap, rstride=rstride, cstride=cstride, linewidth=0, **kw)

    for lab in ulabels:
        fil = labels==lab
        c = [c_/255. for c_ in Tango.hex2rgb(next(colors))]
        ax.scatter(X[fil, :][:, 0], X[fil, :][:, 1], wadX[fil],
                   edgecolor='k', linewidth=.4,
                   c=c, label=lab, marker=next(markers))

    ax.set_zlim(-1.5,1.5)
    mi, ma = Xgrid.min(0), Xgrid.max(0)
    ax.set_xlim(mi[0], ma[0])
    ax.set_ylim(mi[1], ma[1])

    ax.legend(ncol=ncol, loc=0)

    return ax

def plot_waddington_lansdscape(mc, ax=None, resolution=60, xmargin=(.075, .075), ymargin=(.075, .075), cmap='grey'):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    (Xgrid, wadXgrid, X, wadX) = waddington_landscape(mc.gplvm, resolution, xmargin, ymargin)
    r = lambda x: x.reshape(resolution, resolution).T
    CS = ax.contourf(r(Xgrid[:,0]), r(Xgrid[:,1]), r(wadXgrid), linewidths=.6)
    mi, ma = Xgrid.min(0), Xgrid.max(0)
    ax.set_xlim(mi[0], ma[0])
    ax.set_ylim(mi[1], ma[1])

    return CS

def plot_time_graph(mc, labels=None, ulabels=None, start=0, startoffset=(10,5), ax=None, cmap='magma'):
    if labels is None:
        labels = np.zeros(mc.X.shape[0])

    if ulabels is None:
        ulabels = []
        for l in labels:
            if l not in ulabels:
                ulabels.append(l)
        ulabels = np.asarray(ulabels)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    X = mc.X
    
    pseudo_time = mc.get_pseudo_time(start)
    plot_graph_nodes(mc.X, pseudo_time, labels, ulabels, ax, cmap=cmap)

    G = nx.Graph(mc.get_time_graph(start))
    ecols = [e[2]['weight'] for e in G.edges(data=True)]
    cmap = plt.get_cmap(cmap)    
    pos = dict([(i, x) for i, x in zip(range(X.shape[0]), X)])
    edges = nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=ecols, edge_cmap=cmap, width=1)

    cbar = fig.colorbar(edges, ax=ax, pad=.01, fraction=.1, ticks=[], drawedges=False)
    cbar.ax.set_frame_on(False)
    cbar.solids.set_edgecolor("face")
    cbar.set_label('pseudo time')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    #ax.scatter(*X[start].T, edgecolor='red', lw=1.5, facecolor='none', s=50, label='start')
    ax.annotate('start', xy=X[start].T, xycoords='data',
                    xytext=startoffset, textcoords='offset points',
                    size=9,
                    color='.4',
                    bbox=dict(boxstyle="round", fc="0.8", ec='1', pad=.01),
                    arrowprops=dict(arrowstyle="fancy",
                                    fc="0.6", ec="none",
                                    #patchB=el,
                                    connectionstyle="angle3,angleA=17,angleB=-90"),
                    )

    #ax.legend(bbox_to_anchor=(0., 1.02, 1.2, .102), loc=3,
    #           ncol=4, mode="expand", borderaxespad=0.)


def plot_graph_nodes(X, pt, labels, ulabels, ax, cmap='magma', cmap_index=None, box=True, text_kwargs=None, scatter=True, **scatter_kwargs):
    #Tango = GPy.plotting.Tango
    #Tango.reset()
    marker = itertools.cycle('<>sd^')

    if len(ulabels) <= 1:
        if scatter:
            ax.scatter(*X.T, linewidth=.1, c=pt, alpha=.8, edgecolor='w', marker=next(marker), label=None, cmap=cmap)
    else:
        label_pos, col, mi, ma = _get_label_pos(X, pt, labels, ulabels)
        colors = _get_colors(cmap, col, mi, ma, cmap_index)
        for l in ulabels:
            #c = Tango.nextMedium()
            c, r = colors[l]
            fil = (labels==l)
            if scatter:
                ax.scatter(*X[fil].T, linewidth=.1, facecolor=c, alpha=.8, edgecolor='w', marker=next(marker), label=l)
    
            p = label_pos[l]
            rgbc = c#[_c/255. for _c in Tango.hex2rgb(c)]
            if r <.5:
                ec = 'w'
            else:
                ec = 'k'
            if box:
                fc = list(rgbc)
                #fc[-1] = .7
                props = dict(boxstyle='round', facecolor=fc, alpha=0.8, edgecolor=ec)
            else:
                props = dict()
            ax.text(p[0], p[1], l, alpha=.9, ha='center', va='center', color=ec, bbox=props, **text_kwargs or {})

def _get_colors(cmap, col, mi, ma, cmap_index):
    if cmap_index is None:
        cmap = plt.cm.get_cmap(cmap)
        colors = dict([(l, (cmap((col[l]-mi)/(ma-mi)), (col[l]-mi)/(ma-mi))) for l in col])
    else:
        cmap = sns.color_palette(cmap, len(col))[cmap_index]
        r = np.linspace(0,1,len(col))[cmap_index]
        colors = dict([(l, (cmap, r)) for l in col])
    return colors

def _get_sort_dict(labels, ulabels):
    sort_dict = {}#np.empty(labels.shape, dtype=int)
    curr_i = 0
    for i, l in enumerate(ulabels):
        hits = labels==l
        sort_dict[l] = np.where(hits)[0]
        curr_i += hits.sum()
    return sort_dict

def _get_label_pos(X, pt, labels, ulabels):
    sort_dict = _get_sort_dict(labels, ulabels)
    label_pos = {}
    col = {}
    mi, ma = np.inf, 0
    for l in ulabels:
        label_pos[l] = X[sort_dict[l]].mean(0)
        c = pt[sort_dict[l]].mean()
        col[l] = c
        mi = min(mi, c)
        ma = max(ma, c)
    return label_pos, col, mi, ma