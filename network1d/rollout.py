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
import graph1d.generate_normalized_graphs as nz
import numpy as np
import torch as th
import copy
import time

def set_boundary_conditions_dirichlet(matrix, graph, params, bcs, time_index):
    """
    Set boundary conditions to a matrix.

    Arguments:
        matrix: the 2D array
        graph: DGL graph
        params: dictionary of parameters
        bcs: 3D array containing the boundary conditions. 
             dim 1: node index, dim 2: pressure (0) and flow rate (1),
             dim 3: timesteps
        time index (int): index of timestep where we have to take the boundary
                          conditions from

    """
    # set boundary conditions
    if params['bc_type'] == 'realistic_dirichlet':
        matrix[graph.ndata['outlet_mask'].bool(), 0] = bcs[graph.ndata['outlet_mask'].bool(), 0, time_index]
        matrix[graph.ndata['inlet_mask'].bool(), 1] = bcs[graph.ndata['inlet_mask'].bool(), 1, time_index]
    elif params['bc_type'] == 'full_dirichlet':
        matrix[graph.ndata['inlet_mask'].bool(), 0] = bcs[graph.ndata['inlet_mask'].bool(), 0, time_index]
        matrix[graph.ndata['outlet_mask'].bool(), 0] = bcs[graph.ndata['outlet_mask'].bool(), 0, time_index]
        matrix[graph.ndata['inlet_mask'].bool(), 1] = bcs[graph.ndata['inlet_mask'].bool(), 1, time_index]
        matrix[graph.ndata['outlet_mask'].bool(), 1] = bcs[graph.ndata['outlet_mask'].bool(), 1, time_index]

def set_next_flowrate(graph, bcs, time_index):
    """
    Set physiological boundary conditions to a graph.

    Arguments:
        graph: DGL graph
        params: dictionary of parameters
        bcs: 3D array containing the boundary conditions. 
             dim 1: node index, dim 2: pressure (0) and flow rate (1),
             dim 3: timesteps
        time index (int): index of timestep where we have to take the boundary
                          conditions from

    """
    graph.ndata['next_flowrate'] = bcs[:, 1, time_index]

def perform_timestep(gnn_model, params, graph, bcs, time_index, set_bcs = True):
    """
    Performs a single timestep of the rollout phase.

    Arguments:
        gnn_model: the GNN model
        params: dictionary of parameters
        graph: DGL graph
        bcs: 3D array containing the boundary conditions. 
             dim 1: node index, dim 2: pressure (0) and flow rate (1),
             dim 3: timesteps
        time index (int): index of timestep where we have to take the boundary
                          conditions from
        set_bcs (bool): set boundary conditions. Default -> True
    Returns:
        2D array where dim 1 corresponds to node indices, and dim 2 corresponds 
            to pressure (0) and flow rate (1)

    """

    gf = graph.ndata['nfeatures']
    set_next_flowrate(graph, bcs, time_index)
    if 'dirichlet' in params['bc_type']:
        set_boundary_conditions_dirichlet(gf, graph, params, bcs, time_index)

    delta = gnn_model(graph)
    gf[:,0:2] = gf[:,0:2] + delta

    if set_bcs:
        if 'dirichlet' in params['bc_type']:
            set_boundary_conditions_dirichlet(gf, graph, params, bcs,
                                              time_index)
        elif params['bc_type'] == 'physiological':
            gf[graph.ndata['inlet_mask'].bool(), 1] = bcs[graph.ndata['inlet_mask'].bool(), 1, time_index]

    return gf[:,0:2]

def compute_average_branches(graph, flowrate):
    """
    Average flowrate over branch nodes

    Arguments:
        graph: DGL graph
        flowrate: 1D tensor containing nodal flow rate values

    """
    branch_id = graph.ndata['branch_id'].detach().numpy()
    bmax = np.max(branch_id)
    for i in range(bmax + 1):
        idxs = np.where(branch_id == i)[0]
        rflowrate = th.mean(flowrate[idxs])
        flowrate[idxs] = rflowrate

def rollout(gnn_model, params, graph, average_branches = True):
    """
    Performs rollout phase.

    Arguments:
        gnn_model: the GNN
        params: dictionary of parameters
        graph: DGL graph
        average_branches: if Trues, averages flowrate over branch nodes.
                          Default -> True

    Returns:
        2D array of reconstructed features, where dim 1 corresponds to node 
            indices and dim 2 corresponds to pressure (0) and flow rate (1),
        2D array containing normalized pressure and flow rate relative errors
        2D array containing pressure and flow rate relative errors
        2D array containing the difference of reconstructed and actual features
        Relative continuity loss
        Elapsed time in seconds

    """
    gnn_model.eval()
    times = graph.ndata['nfeatures'].shape[2]
    graph = copy.deepcopy(graph)
    true_graph = copy.deepcopy(graph)

    tfc = true_graph.ndata['nfeatures'].clone()
    graph.ndata['nfeatures'] = tfc[:,:,0].clone()
    graph.edata['efeatures'] = true_graph.edata['efeatures'].squeeze().clone()

    r_features = graph.ndata['nfeatures'][:,0:2].unsqueeze(axis = 2).clone()
    start = time.time()
    for it in range(times-1):
        # set loading variable
        graph.ndata['nfeatures'][:,-1] = tfc[:,-1,it]
        gf = perform_timestep(gnn_model, params, graph, tfc, it + 1)

        if average_branches:
            compute_average_branches(graph, gf[:,1])

        graph.ndata['nfeatures'][:,0:2] = gf
        r_features = th.cat((r_features, gf.unsqueeze(axis = 2)), axis = 2)

        # set next conditions to exact for debug
        # graph.ndata['nfeatures'][:,0:2] = tfc[:,0:2,it + 1].clone()

    end = time.time()
    tfc = true_graph.ndata['nfeatures'][:,0:2,:].clone()

    rfc = r_features.clone()

    # we only compute errors on branch nodes
    branch_mask = th.reshape(graph.ndata['branch_mask'],(-1,1,1))
    branch_mask = branch_mask.repeat(1,2,tfc.shape[2])

    # compute error
    tfc = tfc * branch_mask
    rfc = rfc * branch_mask
    diff = tfc - rfc

    errs = th.sum(th.sum(diff**2, dim = 0), dim = 1)
    errs = errs / th.sum(th.sum(tfc**2, dim = 0), dim = 1)
    errs_normalized = th.sqrt(errs)

    tfc[:,0,:] = nz.invert_normalize(tfc[:,0,:], 'pressure', 
                                     params['statistics'], 'features')
    tfc[:,1,:] = nz.invert_normalize(tfc[:,1,:], 'flowrate', 
                                     params['statistics'], 'features')

    rfc[:,0,:] = nz.invert_normalize(rfc[:,0,:], 'pressure', 
                                     params['statistics'], 'features')
    rfc[:,1,:] = nz.invert_normalize(rfc[:,1,:], 'flowrate', 
                                     params['statistics'], 'features')

    diff = tfc - rfc
    errs = th.sum(th.sum(diff**2, dim = 0), dim = 1)
    errs = errs / th.sum(th.sum(tfc**2, dim = 0), dim = 1)
    errs = th.sqrt(errs)

    return r_features.detach().numpy(), errs_normalized.detach().numpy(), \
           errs.detach().numpy(), np.abs(diff.detach().numpy()), end - start

    

    




