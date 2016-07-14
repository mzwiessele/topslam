# topslam

[![Build Status](https://travis-ci.org/mzwiessele/topslam.svg?branch=master)](https://travis-ci.org/mzwiessele/topslam)
[![codecov.io](http://codecov.io/github/mzwiessele/manifold/coverage.svg?branch=master)](http://codecov.io/github/mzwiessele/manifold?branch=master)

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

### Using pip

*Not yet uploaded, publication pending*

`$ pip install topslam`

### From source, developmental versions

You can install GPy in the developmental version, as well as topslam. Recommended is anaconda python.

https://www.continuum.io/downloads

and install numpy, scipy, matplotlib, pandas etc directly through anaconda ($ conda install numpy scipy matplotlib pandas <etc..>)

#### GPy:

```
$ git checkout git@github.com:SheffieldML/GPy.git
$ cd GPy
$ python setup.py develop
```

#### topslam:

```
$ git checkout git@github.com:mzwiessele/topslam.git
$ cd topslam
$ python setup.py develop
```

This should install all necessary packages, even if you did not install them through anaconda (be careful, numpy ans scipy will try to build if you do not use anaconda for it).

#### pull updates:

In the topslam directory you just pull in changes using:

`$ git pull`

This should update it, unless you have made changes.

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

## Example Applications:

See https://github.com/mzwiessele/topslam/tree/master/notebooks and check for updates, as there will be more notebooks added.
