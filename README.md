# topslam

[![Build Status](https://travis-ci.org/mzwiessele/manifold.svg?branch=master)](https://travis-ci.org/mzwiessele/manifold) [![codecov.io](http://codecov.io/github/mzwiessele/manifold/coverage.svg?branch=master)](http://codecov.io/github/mzwiessele/manifold?branch=master)

Extracting and using probabilistic Waddington's landscape recreation from single cell gene expression measurements.

## Citation

    Journal publication pending

    @Misc{topslam2016,
      author =   {{Max Zwiessele}},
      title =    {{topslam}: Probabilistic Epigenetic Landscapes for Single Cell Gene Expression Experiments},
      howpublished = {\url{https://github.com/mzwiessele/topslam}},
      year = {since 2016}
    }


    @Misc{gpy2014,
      author =   {{GPy}},
      title =    {{GPy}: A Gaussian process framework in python},
      howpublished = {\url{http://github.com/SheffieldML/GPy}},
      year = {since 2012}
    }


## Installation

*Not yet uploaded, publication pending*

`$ pip install topslam`

## Data filtering and usage

It is recommended to use pandas to hold your data.

RNAseq experiments suffer from a lot of different sources of noise. Therefore,
we try to filter very harshly to make sure we capture as much signal as possible.
This can also be detrimental to the optimization process and you might
choose to filter differently. All of filtering is strongly dependent on your
data. Thus, know your data and decide whether to use this filtering or
filter yourself for the optimal signal to noise ratio.

This is (empirically in our experience) the optimal way (we found) of using a
selected subset of genes in order to learn a BayesianGPLVM for it.

## Model Opimization

We provide an optimization routine in `topslam.optimize.optimize_model`, which optimizes a model in the way we optimize the models.
This is only a first order help for optimization. If you have trouble optimizing your data, consider z-normalization before optimization.
Additionally we provide a model creator, which creates a BayesianGPLVM model for you. See the example application below for usage.

Remarks for an optimized model for standard normalized data (zero mean and unit variance) after optimization:

  - Gaussian_noise.variance smaller then 0.5
  - rbf.variance value between 0 and 1, but bigger then Gaussian_noise.variance
  - dimensionality of landscape can be seen by m.kern.plot_ARD(), big values are used dimensions.
  - plot_latent does not show Waddington's landscape (as grayscale), but the certainty of the model.
  - plot_magnification shows a representation of Waddington's landscape (as grayscale), although we use a transformation of that to show valleys more finegrained.

## Example Application

```python



```
