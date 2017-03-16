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
# * Neither the name of topslam.optimization nor the names of its
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

from sklearn.manifold import TSNE, SpectralEmbedding, Isomap
from sklearn.decomposition import FastICA, PCA
import numpy as np

methods = {'t-SNE':TSNE(n_components=2, perplexity=50, learning_rate=750, n_iter=2000, init='pca'),
           'PCA':PCA(n_components=2),
           'Spectral': SpectralEmbedding(n_components=2, n_neighbors=20),
           'Isomap': Isomap(n_components=2, n_neighbors=20),
           'ICA': FastICA(n_components=2)
           }

def run_methods(Y, methods):
    order = methods.keys()
    dims = {}
    i = 0
    for name in order:
        method = methods[name]
        q = method.n_components
        dims[name] = slice(i, i+q)
        i += q

    latent_spaces = np.empty((Y.shape[0], i))

    for name in methods:
        method = methods[name]
        try:
            _lat = method.fit_transform(Y)
            latent_spaces[:, dims[name]] = _lat
        except:
            raise
            print("Error detected in running method, ignoring this method as NAN")
    latent_spaces -= latent_spaces.mean(0)
    latent_spaces /= latent_spaces.std(0)
    return latent_spaces, dims


def optimize_model(m):
    """
    Optimization routine we use to optimize a (Bayesian-)GPLVM.

    We first fix the variances to low values to force
    the latent mapping function `f` (`Y = f(X)`) learn
    the structure in the beginning and not allow the
    model to explain the data by pure noise.

    Usage example:

        from .simulation import run_methods
        Y -= Y.mean(0) # Normalization of data, zero mean is usually what you want.
        Y /= Y.std(0) # Beware of your data and decide whether you want to normalize the variances!
        X_init, dims = run_methods(Y, methods)
        m = create_model(Y, X_init, num_inducing=30)
        optimize_model(m)
    """
    m.update_model(False)
    m.likelihood[:] = m.Y.values.var()/10.
    try:
        m.X.variance[:] = .1
    except: #normal GPLVM
        pass

    try:
        m.kern['.*lengthscale'].fix()
    except AttributeError:
        pass
    try:
        m.kern['.*variances'].fix(m.Y.values.var()/1e5)
    except:
        pass

    m.likelihood.fix()
    m.update_model(True)
    m.optimize(max_iters=500, messages=1, clear_after_finish=True)

    m.likelihood.unfix()
    m.optimize(max_iters=500, messages=1, clear_after_finish=True)

    m.kern.unfix()
    m.optimize(max_iters=1e5, messages=1)
    return m

def create_model(Y, X_init=None, num_inducing=10, nonlinear_dims=5, linear_dims=0, white_variance=1):
    """
    Create a BayesianGPLVM model for the expression values in Y.

    Y has the cells on the rows and genes across dimensions:
        Y.shape == (#cells, #genes)

    X_init is the initial latent space for the model.
    Usually this is being initialized by using simulation.run_methods
        X_init, dims = run_methods(Y, methods)

    num_inducing are the number of inducing inputs. It is a number `M`
    between the `0` and the number of datapoints you have and controls
    the complexity of your model. We usually use 10 to 20
    inducing inputs, but if you are having trouble with accuracy in
    your found landscape, you can try to up this number. Note, that
    the speed of the method goes down, with higher numbers of
    inducing inputs. Also, if you use RNASeq data, it is recommended to use a
    lower number (i.e. 10) of inducing inputs so the BayesianGPLVM is
    forced to generalise over patterns and cannot explain the zeros in the
    data by inducing inputs.

    nonlinear_dims are the number of latent dimensions modelled as nonlinear
    relationship between latent space and observed gene expression values
    along the samples. This value gets ignored if X_init is given and the number
    of nonlinear_dims will be the number of dimensions in X_init. If X_init is
    not given, it will be created by PCA.

    linear_dims are the linear dimensions to add into the latent space.
    Linear dimensions are used for modelling linear relationships in the latent
    space independently from the non-linear ones. That is, the last linear_dims
    dimensions in the latent space will be modelled by a linear kernel. We
    recommend try to first run without linear dimensions and see what the
    BayesianGPLVM can learn. If there is a considered amount of confounding
    variation, the linear dimension can help to find this variation
    and explain it away from the rest. It can also lead to unexpected results...

    white_variance is a white variance value (float) for a white variance on the 
    kernel. If it is None, no white variance kernel will be added to the analysis.

    Missing Data: If you have missing data, you can assign the values in Y,
    which are missing to np.nan and the BayesianGPLVM will assume missing
    data at random over those. This will include the dimensionality in
    the runtime of the method and will slow down progress significantly. Thus,
    only include missing data into the model, if you are certain you want to
    use it.

    Usage example:

        from .simulation import run_methods
        Y -= Y.mean(0) # Normalization of data, zero mean is usually what you want.
        Y /= Y.std(0) # Beware of your data and decide whether you want to normalize the variances!
        X_init, dims = run_methods(Y, methods)
        m = create_model(Y, X_init, num_inducing=10)
        optimize_model(m)

    returns a BayesianGPLVM model for the given data matrix Y.
    """
    from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch
    from GPy.kern import Linear, RBF, Add, White
    from GPy.util.linalg import pca

    try:
        Y = Y.values.copy()
    except:
        Y = np.asarray(Y, float).copy()

    if X_init is None:
        X_init = pca(Y, nonlinear_dims)[0]

    kernels = []

    if linear_dims > 0:
        Qlin = linear_dims
        Q = X_init.shape[1] + Qlin
        kernels.extend([
            RBF(Q-Qlin, ARD=True, active_dims=np.arange(0,X_init.shape[1])),
            Linear(Qlin, ARD=True, active_dims=np.arange(X_init.shape[1], Q))
        ])
    else:
        Q = X_init.shape[1]
        kernels.append(RBF(Q, ARD=True, active_dims=np.arange(0,X_init.shape[1])))

    if white_variance is not None:
        kernels.append(White(Q, variance=white_variance))
    
    if len(kernels) > 1:
        kernel = Add(kernels)
    else:
        kernel = kernels[0]
    
    m = BayesianGPLVMMiniBatch(Y, Q, X=X_init,
                    kernel=kernel,
                    num_inducing=num_inducing,
                    missing_data=np.any(np.isnan(Y))
                    )

    return m
