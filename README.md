# applygpy

[![Build Status](https://travis-ci.org/mzwiessele/applygpy.svg?branch=master)](https://travis-ci.org/mzwiessele/applygpy) [![codecov.io](http://codecov.io/github/mzwiessele/applygpy/coverage.svg?branch=master)](http://codecov.io/github/mzwiessele/applygpy?branch=master)

This package is for general reoccuring testing schemes when applying GPs to datasets (Classification and regression so far).

It is easy to run a crossvalidation based model selection:

```
X = np.random.uniform(-1, 1, (200, 1))
k = GPy.kern.Matern32(1)
Y = np.random.multivariate_normal(np.zeros(X.shape[0]), k.K(X))[:,None]
test_models = [
    ['Mat+Lin', kern.Matern32(X.shape[1]) + kern.Linear(X.shape[1], variances=.01) + kern.Bias(X.shape[1])], 
    ['RBF+Lin', kern.Exponential(X.shape[1]) + kern.Linear(X.shape[1], variances=.01) + kern.Bias(X.shape[1])],
    ['Lin', kern.Linear(X.shape[1], variances=.01) + kern.Bias(X.shape[1])],
] 
```

## Regression:
```
res = cross_validate(X, Y, verbose=False, 
                     kernels_models=test_models,
                     k=2,
                     #model_builder=model_builder
                    )
```

## Classification:
```
res = cross_validate(X, Y>Y.mean(), verbose=False, 
                     kernels_models=test_models,
                     k=2,
                     #model_builder=model_builder
                    )
```

res has two errors (log-likelihood and RMSE) measures and the number of samples in the filds of the crossvalidation.

