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