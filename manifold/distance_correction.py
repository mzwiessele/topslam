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
# * Neither the name of paramz.core.constrainable nor the names of its
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

from manifold import distances
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.cluster.hierarchy import average, fcluster, dendrogram
from scipy.spatial.distance import pdist
from Bio.Phylo.PhyloXML import Point

class ManifoldCorrection(object):
    def __init__(self, gplvm, distance=distances.mean_embedding_dist):
        """
        Construct a correction class for the BayesianGPLVM given.

        All evaluations on this object are lazy, so do not change attributes
        at runtime in order to have a consistent model.

        :param [GPy.models.BayesianGPLVM,GPy.models.GPLVM] gplvm:
            an optimized GPLVM or BayesianGPLVM model from GPy
        :param func dist: dist(X,G), the distance to use for pairwise distances
            in X using the manifold embedding G
        """
        self.gplvm = gplvm
        self.distance = distance

    @property
    def X(self):
        if getattr(self, '_X', None) is None:
            try:
                self._X = self.gplvm.X.mean
                self._X.mean
            except AttributeError:
                # not bayesian GPLVM
                self._X = self.gplvm.X
        return self._X

    @property
    def G(self):
        if getattr(self, '_G', None) is None:
            self._G = self.gplvm.predict_wishard_embedding(self.X)
        return self._G

    @property
    def _manifold_distance_matrix(self):
        if getattr(self, '_M', None) is None:
            self._M = self.distance(self.X, self.G)
        return self._M

    @property
    def minimal_spanning_tree(self):
        """
        Create a minimal spanning tree using the distance correction method given.

        You can explore different distance corrections in manifold.distances.
        """
        if getattr(self, '_mst', None) is None:
            self._mst = minimum_spanning_tree(self._manifold_distance_matrix)
        return self._mst

    @property
    def manifold_corrected_distances(self):
        """
        Return the distances summed along the manifold minimal spanning tree.

        This reflects the structure and distances along the manifold in order
        to include the pseudo stretch of the Wishart embedding of the manifold.
        """
        if getattr(self, '_corrected_distances', None) is None:
            self._corrected_distances = dijkstra(self.minimal_spanning_tree, directed=False, return_predecessors=True)
        return self._corrected_distances[0]

    @property
    def manifold_corrected_structure(self):
        """
        Return the structure distances, where each edge along the manifold
        minimal spanning tree has a distance of one, such that the
        distance just means the number of hops to make in order to get from
        one point to another. This can be very helpful in doing structure analysis
        and clustering of the manifold embedded data points.
        """
        if getattr(self, '_corrected_distances', None) is None:
            self._corrected_structure = dijkstra(self.minimal_spanning_tree, directed=False, unweighted=True)
        return self._corrected_structure


    def manifold_structure_linkage(self):
        """
        Return the UPGMA linkage matrix based on the correlation structure of
        the manifold embedding MST
        """
        if getattr(self, '_Slinkage', None) is None:
            self._Slinkage = average(pdist(self.manifold_corrected_structure, metric='correlation'))
        return self._Slinkage

    @property
    def manifold_distance_linkage(self):
        """
        Return the UPGMA linkage matrix for the manifold distances
        """
        if getattr(self, '_Mlinkage', None) is None:
            self._Mlinkage = average(self.manifold_corrected_distances)
        return self._Mlinkage

    def cluster(self, linkage, num_classes):
        """
        Cluster the linkage matrix into num_classes number of classes.
        """
        return fcluster(linkage, t=num_classes, criterion='maxclust')

    def plot_dendrogram(self, linkage, **kwargs):
        """
        plot a dendrogram for the linkage matrix with leaf labels. The kwargs go
        directly into the scipy function :py:func:`scipy.cluster.hierarchy.dendrogram`
        """
        return dendrogram(linkage, **kwargs)

    def pseudo_time(self, start):
        """
        Returns the pseudo times along the manifold for the given starting point
        start to all other points (including start).

        :param int start: The index of the starting point in self.X
        """
        return self.manifold_corrected_distances[start]