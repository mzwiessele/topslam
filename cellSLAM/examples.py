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
# * Neither the name of cellSLAM nor the names of its
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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
#import pygraphviz

import pods, pandas as pd, os
import GPy, numpy as np
from GPy.models import BayesianGPLVM

def example_deng():
    # This is the process of how we loaded the data:
    ulabels = ['Zygote', 
               '2-cell embryo', 
               'Early 2-cell blastomere', 'Mid 2-cell blastomere', 'Late 2-cell blastomere',
               '4-cell blastomere', '8-cell blastomere', '16-cell blastomere',
               'Early blastocyst cell', 'Mid blastocyst cell', 'Late blastocyst cell',
               'fibroblast', 
               'adult liver',
              ]
    
    folder_path = 'Deng'
    csv_file = os.path.join(folder_path, 'expression_values.csv')
    if os.path.exists(csv_file):
        Ydata = pd.read_csv(csv_file, index_col=[0,1,2], header=0)
    else:
        data = GPy.util.datasets.singlecell_rna_seq_deng()
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        Ydata = data['Y'].copy()
        Ydata.columns = Ydata.columns.to_series().apply(str.upper)
        Ydata = Ydata.reset_index().set_index('index', append=True)
        Ydata['labels'] = data['labels'].values
        Ydata = Ydata.set_index('labels', append=True)
        Ydata = Ydata.reorder_levels([0,2,1])
        Ydata = Ydata.reset_index([0,2]).loc[ulabels].set_index(['level_0', 'index'], append=True)
        Ydata.to_csv(csv_file)