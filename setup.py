#!/usr/bin/env python
# -*- coding: utf-8 -*-

#===============================================================================
# Copyright (c) 2015, Max Zwiessele
#
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
# * Neither the name of paramax nor the names of its
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

from __future__ import print_function
import os
from setuptools import setup
import codecs

def read(fname):
    with codecs.open(fname, 'r', 'latin') as f:
        return f.read()

def read_to_rst(fname):
    try:
        import pypandoc
        rstname = "{}.{}".format(os.path.splitext(fname)[0], 'rst')
        pypandoc.convert(read(fname), 'rst', format='md', outputfile=rstname)
        with open(rstname, 'r') as f:
            rststr = f.read()
        return rststr
        #return read(rstname)
    except ImportError:
        return read(fname)

desc = read('README.md')

version_dummy = {}
exec(read('topslam/__version__.py'), version_dummy)
__version__ = version_dummy['__version__']
del version_dummy

setup(name = 'topslam',
      version = __version__,
      author = "Max Zwiessele",
      author_email = "ibinbei@gmail.com",
      description = ("topslam metric and correction techniques for (Bayesian) GPLVM"),
      long_description=desc,
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "https://github.com/mzwiessele/topslam",
      #packages = ["cellSLAM",
      #            "cellSLAM/tests"
      #            "cellSLAM/simulation"
      #            ],
      package_dir={'topslam': 'topslam'},
      py_modules = ['topslam.__init__'],
      #test_suite = 'cellSLAM.tests',
      install_requires=['GPy>=1', 'scikit-learn', 'pandas', 'pods', 'seaborn', 'adjustText'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
