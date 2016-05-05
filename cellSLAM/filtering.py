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
# * Neither the name of cellSLAM.filtering nor the names of its
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

import numpy as np

def filter_RNASeq(E, transform_log1p=True):
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