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
# * Neither the name of topslam nor the names of its
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
from topslam.plotting import plot_comparison
from topslam.pseudo_time.tree_correction import ManifoldCorrectionTree


def example_deng(optimize=True, plot=True):
    import pandas as pd, os
    import GPy, numpy as np
    from topslam.filtering import filter_RNASeq
    # Reproduceability, BGPLVM has local optima
    np.random.seed(42)

    # This is the process of how we loaded the data:
    ulabels = ['Zygote',
               '2-cell embryo',
               'Early 2-cell blastomere', 'Mid 2-cell blastomere', 'Late 2-cell blastomere',
               '4-cell blastomere', '8-cell blastomere', '16-cell blastomere',
               'Early blastocyst cell', 'Mid blastocyst cell', 'Late blastocyst cell',
               'fibroblast',
               'adult liver',
              ]

    folder_path = os.path.expanduser('~/tmp/Deng')
    csv_file = os.path.join(folder_path, 'filtered_expression_values.csv')

    if os.path.exists(csv_file):
        print('Loading previous filtered data: {}'.format(csv_file))
        Y_bgplvm = pd.read_csv(csv_file, index_col=[0,1,2], header=0)
    else:
        print('Loading data:')
        data = GPy.util.datasets.singlecell_rna_seq_deng()
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        Ydata = data['Y'].copy()
        Ydata.columns = Ydata.columns.to_series().apply(str.upper)
        Ydata = Ydata.reset_index().set_index('index', append=True)
        Ydata['labels'] = data['labels'].values
        Ydata = Ydata.set_index('labels', append=True)
        Ydata = Ydata.reorder_levels([0,2,1])
        Ydata = Ydata.reset_index([0,2]).loc[ulabels].set_index(['level_0', 'index'], append=True)

        Y = Ydata.copy()
        Y.columns = [c.split('.')[0] for c in Y.columns]
        Y_bgplvm = filter_RNASeq(Y)
        print('\nSaving data to tmp file: {}'.format(csv_file))
        Y_bgplvm.to_csv(csv_file)

    labels = Y_bgplvm.index.get_level_values(0).values
    Ymean = Y_bgplvm.values.mean()
    Ystd = Y_bgplvm.values.std()

    Y_m = Y_bgplvm.values
    Y_m -= Ymean
    Y_m /= Ystd

    # get the labels right for split experiments
    # get the labels right for 8 and split
    new_8_labels = []
    for _l in Y_bgplvm.loc['8-cell blastomere'].index.get_level_values(1):
        _l = _l.split('-')[0]
        if not('split' in _l):
            new_8_labels.append('8')
        elif not('pooled' in _l):
            new_8_labels.append('8 split')
        else:
            new_8_labels.append('8 split')

    labels[labels=='8-cell blastomere'] = new_8_labels

    # get the labels right for 16 and split
    new_16_labels = []
    for _l in Y_bgplvm.loc['16-cell blastomere'].index.get_level_values(1):
        _l = _l.split('-')[0]
        if not('split' in _l):
            new_16_labels.append('16')
        elif not('pooled' in _l):
            new_16_labels.append('16 split')
        else:
            new_16_labels.append('16 split')

    labels[labels=='16-cell blastomere'] = new_16_labels

    ulabels = []
    for lab in labels:
        if lab not in ulabels:
            ulabels.append(lab)

    short_labels = labels.copy()
    _ulabels_convert = np.array([
            'Z',# Z',
            'E',# Em',
            '2',# Bm E',
            '2',# Bm M',
            '2',# Bm L',
            '4',
            '8',
            '8 s',
            '16',
            '16 s',
            'Bz',# E',
            'Bz',# M',
            'Bz',# L'
            'F',
            'L'
        ])

    short_ulabels = []
    for lab, nlab in zip(ulabels, _ulabels_convert):
        short_labels[short_labels==lab] = nlab
        if nlab not in short_ulabels:
            short_ulabels.append(nlab)

    from topslam.optimization import run_methods, methods, create_model, optimize_model
    X_init, dims = run_methods(Y_m, methods)

    m = create_model(Y_m, X_init, num_inducing=25)
    m.Ymean = Ymean
    m.Ystd = Ystd
    m.data_labels = short_labels
    m.data_ulabels = short_ulabels
    m.data = Y_bgplvm

    m.X_init = X_init
    m.dims = dims

    if optimize:
        optimize_model(m)
    if plot:
        mc = ManifoldCorrectionTree(m)
        plot_comparison(mc, X_init, dims, m.data_labels, m.data_ulabels, 0)

    return m




