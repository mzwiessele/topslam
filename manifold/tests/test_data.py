'''
Created on 30 Sep 2015

@author: Max Zwiessele
'''
import unittest, numpy as np, GPy  # @UnresolvedImport
from io import BytesIO
from manifold import distance_correction

class Test(unittest.TestCase):


    def setUp(self):
        np.random.seed(11111)
        self.X = np.linspace(-1, 1, 20)[:,None]
        k = GPy.kern.Matern32(1, lengthscale=1, variance=1)
        self.sim_model = 'Mat+Lin'
        self.mf = GPy.mappings.Linear(1, 1)
        self.mf[:] = .01
        self.mu = self.mf.f(self.X)
        self.Y = np.random.multivariate_normal(np.zeros(self.X.shape[0]), k.K(self.X))[:,None]

    def testLibSVM(self):
        from sklearn.datasets import dump_svmlight_file  # @UnresolvedImport
        i = BytesIO()
        dump_svmlight_file(self.X, self.Y.flat, i)
        i.seek(0)
        X, Y = load_libsvm(i)
        np.testing.assert_allclose(X.toarray(), self.X)
        np.testing.assert_allclose(Y, self.Y.flat)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCrossval']
    unittest.main()