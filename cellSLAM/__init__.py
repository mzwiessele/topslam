from .tree_correction import ManifoldCorrectionTree
from .knn_correction import ManifoldCorrectionKNN
from .waddington_landscape import waddington_landscape, plot_waddington_landscape
from .optimization import optimize_model

import distances

import simulation


def filter_RNASeq(dataframe):
    import numpy as np
    Y = dataframe.copy()
    print('Before filtering: #cells={}, #genes={}'.format(*Y.shape))

    # omit cells with no coverage
    fil = (Y.sum(1)>0).values
    Y = Y.loc[fil, :]

    # omit genes with overrepresentation of zeros:
    fil = ((Y==0).sum(0)/Y.shape[0])<.1
    Y = Y.ix[:, fil]

    # omit genes with too low variance
    var = Y.var()
    var[np.isnan(var)] = 0
    perc = np.percentile(var.values.flat, 90)
    fil = (var>perc).values
    Y = Y.loc[:, fil]

    # Drop any missing values for ease of use
    Y = Y.dropna(axis=1)

    labels = Y.index.get_level_values(0)

    Y = np.log1p(Y)

    Ymean = Y.mean()
    Ystd = Y.std()

    Y -= Ymean
    Y /= Ystd

    print('After filtering: #cells={}, #genes={}'.format(*Y.shape))
    return Y, labels, Ymean, Ystd