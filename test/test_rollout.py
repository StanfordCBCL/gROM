# Copyright 2023 Stanford University

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

import sys
import os
sys.path.append(os.getcwd())
import torch as th
import graph1d.generate_normalized_graphs as gng
import graph1d.generate_dataset as dset
import tools.io_utils as io
from network1d.meshgraphnet import MeshGraphNet
from network1d.tester import get_gnn_and_graphs
from network1d.rollout import rollout
import math

if __name__ == "__main__":
    path = 'test/test_data/gnn_model'
    data_location = 'test/test_data/'
    graphs_folder = 'graphs/'
    gnn_model, graphs, params = get_gnn_and_graphs(path, graphs_folder, 
                                                   data_location)
    graph_name = 's0095_0001.0.3.grph'
    _, _, err, _, _ = rollout(gnn_model, params, graphs[graph_name])
    
    tol = 1e-4
    # check pressure error
    if not math.isclose(err[0], 0.00617872, rel_tol = tol):
        raise ValueError('Incorrect pressure error')

    # check flow rate error
    if not math.isclose(err[1], 0.01505195, rel_tol = tol):
        raise ValueError('Incorrect flow rate error')
    