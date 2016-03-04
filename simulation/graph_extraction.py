
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.sparse.lil import lil_matrix
from scipy.sparse.extract import find

def nearest_neighbor_graph(D, k):
    idxs = np.argsort(D)
    r = range(D.shape[0])
    idx = idxs[:, :k]
    _distances = sparse.lil_matrix(D.shape)
    for neighbours in idx.T:
        _distances[r, neighbours] = D[r, neighbours]
    return _distances

def extract_manifold_distances_mst(D):
    # First do the minimal spanning tree distances
    mst = minimum_spanning_tree(D)
    return dijkstra(mst, directed=False, return_predecessors=False), mst

def extract_manifold_distances_knn(D, knn=[3,4,5,7,10], add_mst=None):
    # K Nearest Neighbours distances
    idxs = np.argsort(D)
    r = range(D.shape[0])
    for k in knn:
        idx = idxs[:, :k]
        _distances = sparse.lil_matrix(D.shape)
        for neighbours in idx.T:
            _distances[r, neighbours] = D[r, neighbours]
        if add_mst is not None:
            for i,j,v in zip(*sparse.find(add_mst)):
                if _distances[i,j] == 0:
                    _distances[i,j] = v
        nearest_neighbour_distances = dijkstra(_distances, directed=False)
        yield nearest_neighbour_distances, _distances

def extract_distance_graph(manifold_distance, graph, start):
    """
    Extract a graph with edges filled with the distance from start.
    This is mainly for plotting purposes in order to plot the graph,
    and distances along it.
    """
    pt_graph = lil_matrix(graph.shape)
    D = manifold_distance

    start = 6

    for i,j in zip(*find(graph)[:2]):
        pt_graph[i,j] = D[start,j]
        if j == start:
            pt_graph[i,j] = D[i,start]
    return pt_graph