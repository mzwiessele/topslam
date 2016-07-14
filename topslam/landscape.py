from scipy.special import expit
import numpy as np

def transform_to_wad(X, mu, std, steepness=2):
    test_wad_trace = X.copy()
    test_wad_trace -= mu
    test_wad_trace /= std/steepness
    return (expit(test_wad_trace)-.5)

def waddington_landscape(m, dimensions=None, resolution=60, xmargin=(.075, .075), ymargin=(.075, .075)):
    """
    Extract Waddington's landscape from a (Bayesian-)GPLVM model `m`.
    The landscape surface is extracted using a grid in the inputs with
    size [resolution x resolution].

    returns Xgrid, wadXgrid, X, wadX
        - Xgrid is the grid made for predicting the surface wadXgrid.
        - X is the used dimensions of the input of the (B)GPLVMs surface wadX at those points.
    """
    if dimensions is None:
        msi = m.get_most_significant_input_dimensions()[:2]
    else:
        msi = dimensions[:2]

    X = m.X.mean.values.copy()
    X[:, np.setdiff1d(range(X.shape[1]), msi)] = 0.

    [xmin, ymin], [xmax, ymax] = X[:, msi].min(0), X[:, msi].max(0)
    rx, ry = xmax-xmin, ymax-ymin

    xmin = xmin - xmargin[0]*rx
    xmax = xmax + xmargin[1]*rx

    ymin = ymin - ymargin[0]*ry
    ymax = ymax + ymargin[1]*ry

    xx, yy = np.mgrid[xmin:xmax:1j*resolution,ymin:ymax:1j*resolution]

    Xgrid = np.c_[xx.T.flat, yy.T.flat]

    Xpred = np.zeros((Xgrid.shape[0], m.X.shape[1]))
    Xpred[:, msi] = Xgrid

    G = m.predict_magnification(Xpred, dimensions=msi)
    wad_mag = G
    magmean = wad_mag.mean()
    magstd = wad_mag.std()

    GX = m.predict_magnification(X, dimensions=msi)

    return (Xgrid, transform_to_wad(wad_mag, magmean, magstd, steepness=2),
            X[:, msi], transform_to_wad(GX, magmean, magstd, steepness=2))

