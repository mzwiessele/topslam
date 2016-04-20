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
from . import distances

from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.sparse import csr_matrix, find, lil_matrix
from scipy.cluster.hierarchy import average, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform

import numpy as np

class ManifoldCorrection(object):
    def __init__(self, gplvm, distance=distances.mean_embedding_dist, dimensions=None):
        """
        Construct a correction class for the BayesianGPLVM given.

        All evaluations on this object are lazy, so do not change attributes
        at runtime in order to have a consistent model.

        :param [GPy.models.BayesianGPLVM,GPy.models.GPLVM] gplvm:
            an optimized GPLVM or BayesianGPLVM model from GPy
        :param func dist: dist(X,G), the distance to use for pairwise distances
            in X using the cellSLAM embedding G
        :param array-like dimensions: The dimensions of the latent space to use [default: self.gplvm.get_most_significant_input_dimensions()[:2]]
        """
        self.gplvm = gplvm
        self.distance = distance
        if dimensions is None:
            dimensions = self.gplvm.get_most_significant_input_dimensions()[:2]
        self.dimensions = dimensions

    @property
    def X(self):
        if getattr(self, '_X', None) is None:
            try:
                _X = self.gplvm.X.mean
                _X.mean
            except AttributeError:
                # not bayesian GPLVM
                _X = self.gplvm.X
            # Make sure we only take the dimensions we want to use:
            self._X = np.zeros(_X.shape)
            msi = self.dimensions
            self._X[:, msi] = _X[:,msi]
        return self._X

    @property
    def G(self):
        if getattr(self, '_G', None) is None:
            self._G = self.gplvm.predict_wishard_embedding(self.X)
        return self._G

    @property
    def manifold_corrected_distance_matrix(self):
        """
        Returns the distances between all pairs of inputs, corrected for
        the cellSLAM embedding.
        """
        if getattr(self, '_M', None) is None:
            self._M = csr_matrix(self.distance(self.X, self.G))
        return self._M

    @property
    def minimal_spanning_tree(self):
        """
        Create a minimal spanning tree using the distance correction method given.

        You can explore different distance corrections in cellSLAM.distances.
        """
        if getattr(self, '_mst', None) is None:
            self._mst = minimum_spanning_tree(self.manifold_corrected_distance_matrix)
        return self._mst

    @property
    def graph(self):
        """
        Return the correction graph to use for this cellSLAM correction object.
        """
        raise NotImplemented("Implement the graph extraction property for this class")

    def _prep_distances(self):
        self._graph_distances, self._predecessors = dijkstra(self.graph, directed=False, return_predecessors=True)

    @property
    def distances_along_graph(self):
        """
        Return all distances along the graph.

        :param knn_graph: The sparse matrix encoding the knn-graph to compute distances on.
        :param bool return_predecessors: Whether to return the predecessors of each node in the graph, this is for reconstruction of paths.
        """
        if getattr(self, '_graph_distances', None) is None:
            self._prep_distances()
        return self._graph_distances

    @property
    def graph_predecessors(self):
        """
        Return the predecessors of each node for this graph correction.

        This is used for path reconstruction along this graphs shortest paths.
        """
        if getattr(self, '_predecessors', None) is None:
            self._prep_distances()
        return self._predecessors

    @property
    def linkage_along_graph(self):
        """
        Return the UPGMA linkage matrix for the distances along the graph.
        """
        if getattr(self, '_dist_linkage', None) is None:
            self._dist_linkage = average(squareform(self.distances_along_graph))
        return self._dist_linkage

    @property
    def distances_in_structure(self):
        """
        Return the structure distances, where each edge along the graph has a
        distance of one, such that the distance just means the number of
        hops to make in order to get from one point to another.

        This can be very helpful in doing structure analysis
        and clustering of the cellSLAM embedded data points.

        returns hops, the pairwise number of hops between points along the tree.
        """
        if getattr(self, '_struct_distances', None) is None:
            self._struct_distances = dijkstra(self.graph, directed=False, unweighted=True, return_predecessors=False)
        return self._struct_distances

    @property
    def linkage_in_structure(self):
        """
        Return the UPGMA linkage matrix based on the correlation structure of
        the cellSLAM embedding MST
        """
        if getattr(self, '_struct_linkage', None) is None:
            self._struct_linkage = average(pdist(self.distances_in_structure, metric='correlation'))
        return self._struct_linkage

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

    def get_time_graph(self, start):
        """
        Returns a graph, where all edges are filled with the distance from
        `start`. This is mostly for plotting purposes, visualizing the
        time along the tree, starting from `start`.
        """
        test_graph = csr_matrix(self.graph.shape)
        D = self.distances_along_graph
        for i,j in zip(*find(self.graph)[:2]):
            test_graph[i,j] = D[start,j]
            if j == start:
                test_graph[i,j] = D[i,start]
        return test_graph

    def get_longest_path(self, start, report_all=False):
        """
        Get the longest path from start ongoing. This usually coincides with the
        backbone of the tree, starting from the starting point. If the latent
        structure divides into substructures, this is either of the two (if
        two paths have the same lengths). If report_all is True, we find all backbones
        with the same number of edges.
        """
        S = self.distances_in_structure
        preds = self.graph_predecessors
        distances = S[start]
        maxdist = S[start].max()
        ends = (S[start]==maxdist).nonzero()[0]
        paths = []
        for end in ends:
            pre = end
            path = []
            while pre != start:
                path.append(pre)
                pre = preds[start,pre]
            path.append(start)
            if not report_all:
                return path[::-1]
            else:
                paths.append(path[::-1])
        return paths