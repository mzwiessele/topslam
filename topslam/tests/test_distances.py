'''
Created on 31 May 2016

@author: maxz
'''
import unittest, numpy as np
from topslam.pseudo_time import distances

class Test(unittest.TestCase):


    def testPairwise(self):
        X = np.random.normal(0,1,(20,3))
        np.testing.assert_equal(distances.distance_matrix(X), X[:,None]-X[None,:])

    def testMean(self):
        N, Q = 20, 5

        X = np.random.normal(0,1,(N,Q))

        G = np.random.uniform(.5,2,(N,Q,Q))
        G += np.swapaxes(G, 1,2)

        corrD = distances.mean_embedding_dist(X, G)

        i,j = np.random.choice(X.shape[0], size=2, replace=False)

        Dij = X[i]-X[j]
        corrDij = np.sqrt(np.dot(Dij, np.dot(np.mean(np.r_[G[[i]], G[[j]]], 0), Dij)))
        np.testing.assert_almost_equal(corrDij, corrD[i,j])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPairwise']
    unittest.main()