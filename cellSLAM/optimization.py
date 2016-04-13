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
# * Neither the name of cellSLAM.optimization nor the names of its
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
from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch
import numpy as np

methods = {'t-SNE':TSNE(n_components=2),
           'PCA':PCA(n_components=2),
           'Spectral': SpectralEmbedding(n_components=2, n_neighbors=10),
           'Isomap': Isomap(n_components=2, n_neighbors=10),
           'ICA': FastICA(n_components=2)
           }

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
    m.X.variance[:] = .1
    m.kern.lengthscale.fix()
    m.likelihood.fix()
    m.update_model(True)
    m.optimize(max_iters=500, messages=1, clear_after_finish=True)

    m.likelihood.unfix()
    m.optimize(max_iters=500, messages=1, clear_after_finish=True)

    m.kern.lengthscale.unfix()
    m.optimize(max_iters=1e5, messages=1)
    return m
    
def create_model(Y, X_init, num_inducing=25):
    """
    Create a BayesianGPLVM model for the expression values in Y.
    
    Y has the cells on the rows and genes across dimensions:
        Y.shape == (#cells, #genes)
    
    X_init is the initial latent space for the model.
    Usually this is being initialized by using simulation.run_methods
        X_init, dims = run_methods(Y, methods)
        
    num_inducing are the number of inducing inputs. It is a number `M`
    between the `0` and the number of datapoints you have and controls
    the complexity of your model. We usually use between 25 and 50 
    inducing inputs, but if you are having trouble with accuracy in 
    your found landscape, you can try to up this number. Note, that
    the speed of the method goes down, with higher numbers of 
    inducing inputs. 
    
    Usage example:
    
        from .simulation import run_methods
        Y -= Y.mean(0) # Normalization of data, zero mean is usually what you want.
        Y /= Y.std(0) # Beware of your data and decide whether you want to normalize the variances!
        X_init, dims = run_methods(Y, methods)
        m = create_model(Y, X_init, num_inducing=30)
        optimize_model(m)
        
    returns a BayesianGPLVM model for the given data matrix Y. 
    """
    try:
        Y = Y.values
    except:
        pass
    return BayesianGPLVMMiniBatch(Y, X_init.shape[1], X=X_init, num_inducing=num_inducing, missing_data=np.any(np.isnan(Y)))

def filter_data(E, transform_log1p=True):
    """
    Prefilter a pandas DataFrame E [#cells x #genes] of an RNAseq experiment.
    
    RNAseq experiments suffer from a lot of different sources of noise. Therefore, 
    we try to filter very harshly to make sure we capture as much signal as possible.
    This can also be detrimental to the optimization process and you might 
    choose to filter differently. All of filtering is strongly dependent on your
    data. Thus, know your data and decide whether to use this filtering or 
    filter yourself for the optimal signal to noise ratio.  
    
    This is (empirically in our experience) the optimal way (we found) of using a 
    selected subset of genes in order to learn a BayesianGPLVM for it.
    
    transform_log1p decides whether to transform the data after filtering (Y = log(E+1)) 
    
    returns the filtered DataFrame Y
    """
    
    Y = E.copy()
    print ("Before filtering: #cells={} #genes={}".format(*Y.shape))

    # take only the cellcycle genes, for maximum coverage of time line
    # Take cellcycle genes from Macosko et al. 2015
    # converted by the DAVID (from OFFICIAL_GENE_SYMBOL to ENSEMBL_GENE_ID)
    #cellcycle_conversion = pd.read_csv('Trapnell/MacoskoCCConversionENS.txt', sep='\t')
    #cellcycle_filter = np.intersect1d(cellcycle_conversion.From.apply(str.upper), Y.columns.to_series().apply(str.upper))
    #Y = Y[cellcycle_filter]
    
    # omit cells with no coverage
    fil = (Y.sum(1)>0).values
    Y = Y.loc[fil, :]
    
    # omit genes with overrepresentation of zeros:
    fil = ((Y==0).sum(0)/Y.shape[0])<.1
    Y = Y.ix[:, fil]
    
    # omit genes with too low variance
    var = Y.var(0)
    var[np.isnan(var)] = 0
    perc = np.percentile(var.values.flat, 90)
    fil = (var>perc).values
    Y = Y.loc[:, fil]
    
    # Drop any missing values for ease of use
    Y = Y.dropna(axis=1)
    
    n_data, p_genes = Y.shape
    print ("After filtering: #cells={} #genes={}".format(n_data, p_genes))
    
    if transform_log1p:
        print ("Transforming the data Y = log(E + 1)")
        return np.log1p(Y)
    return Y