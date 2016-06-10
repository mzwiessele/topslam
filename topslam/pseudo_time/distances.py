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


import GPy
import numpy as np

def distance_matrix(X):
    """
    Compute the pairwise distance matrix of X with itself.
    """
    return (X[:,None]-X[None, :])

# def _distance_G(dM, G):
#     tmp = GPy.util.linalg.tdot(G)
#     sdist = np.einsum('ijp,ijq,ipqj->ij', dM, dM, tmp)
#     return sdist#, 1./2.)
#
# def _distance(dM):
#     return np.einsum('ijp,ijp->ij', dM, dM)#, 1./2.)
#
# def _multi_chol(G):
#     chols = np.empty(G.shape)
#     for i in range(G.shape[0]):
#         chols[i] = GPy.util.linalg.jitchol(G[i])
#     return chols
#
# def cholesky_dist(X, G):
#     """
#     first product cholesky on vector, then take distance
#     """
#     chols = _multi_chol(G)
#     distM = np.einsum('iq,iqp->ip',X,chols)
#     return np.power(_distance(distance_matrix(distM)), 1./2.)
#
# def cholesky_dist_product(X, G):
#     """first take distance, then product onto cholesky"""
#     chols = _multi_chol(G)
#     return np.power(np.abs(_distance_G((distance_matrix(X)), chols)), 1./2.)

def mean_embedding_dist(X, G):
    """
    The mean correction, correcting distances using the mean between both
    manifold metrics to correct for pairwise distances in X.
    """
    dM = distance_matrix(X)
    mean_geometry = (G[:, None, :, :] + G[None, :, :, :])/2.
    return np.sqrt(np.einsum('ijp,ijpq,ijq->ij', dM, mean_geometry, dM))