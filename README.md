# cellSLAM

[![Build Status](https://travis-ci.org/mzwiessele/manifold.svg?branch=master)](https://travis-ci.org/mzwiessele/manifold) [![codecov.io](http://codecov.io/github/mzwiessele/manifold/coverage.svg?branch=master)](http://codecov.io/github/mzwiessele/manifold?branch=master)

Extracting and using probabilistic Waddington's landscape recreation from single cell gene expression measurements.

## Installation

`$ pip install cellSLAM`

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

We provide an optimization routine in `cellSLAM.optimize.optimize_model`, which optimizes a model in the way we optimize the models. 
This is only a first order help for optimization. If you have trouble optimizing your data, consider z-normalization before optimization. 
Additionally we provide a model creator, which creates a BayesianGPLVM model for you. See the example application below for usage.

## Example Application

```python



```
