from scipy.special import expit
import numpy as np

def transform_to_wad(X, mu, std, steepness=2):
    test_wad_trace = X.copy()
    test_wad_trace -= mu
    test_wad_trace /= std/steepness
    return (expit(test_wad_trace)-.5)

def waddington_landscape(m, resolution=60, xmargin=(.075, .075), ymargin=(.075, .075)):
    """
    Extract Waddington's landscape from a (Bayesian-)GPLVM model `m`.
    The landscape surface is extracted using a grid in the inputs with
    size [resolution x resolution].

    returns Xgrid, wadXgrid, X, wadX
        - Xgrid is the grid made for predicting the surface wadXgrid.
        - X is the used dimensions of the input of the (B)GPLVMs surface wadX at those points.
    """
    msi = m.get_most_significant_input_dimensions()[:2]
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

def plot_waddington_landscape(Xgrid, wadXgrid, X, wadX, ulabels, labels, resolution=60, ncol=5, cmap='terrain', cstride=1, rstride=1, **kw):
    """
    Plot a waddngton landscape with data in 3D.
    Xgrid and wad are the landscape (surface plot, [resolution x resolution])
    and X and wadX are the datapoints as returned by
    Xgrid, wadXgrid, X, wadX = waddington_landscape(m).

    ulabels and labels are the unique labels and labels for each datapoint of X.
    ncol defines the number of columns in the legend above the plot.

    Returns the 3d axis instance of mplot3d.
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(4.66666666655,3.5), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    from GPy.plotting import Tango
    Tango.reset()

    from itertools import cycle
    colors = cycle(Tango.mediumList)
    markers = cycle('<>^vsd')

    r = lambda x: x.reshape(resolution, resolution).T
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