import sys
import os
sys.path.append(os.getcwd())
import graph1d.generate_normalized_graphs as nz
import numpy as np
import torch as th
import copy
import time

def perform_timestep(gnn_model, params, graph, bcs, time_index):
    gf = graph.ndata['nfeatures']

    # set boundary conditions
    if params['bc_type'] == 'realistic_dirichlet':
        gf[graph.ndata['outlet_mask'].bool(), 0] = bcs[graph.ndata['outlet_mask'].bool(), 0, time_index]
        gf[graph.ndata['inlet_mask'].bool(), 1] = bcs[graph.ndata['inlet_mask'].bool(), 1, time_index]
    elif params['bc_type'] == 'full_dirichlet':
        gf[graph.ndata['inlet_mask'].bool(), 0] = bcs[graph.ndata['inlet_mask'].bool(), 0, time_index]
        gf[graph.ndata['outlet_mask'].bool(), 0] = bcs[graph.ndata['outlet_mask'].bool(), 0, time_index]
        gf[graph.ndata['inlet_mask'].bool(), 1] = bcs[graph.ndata['inlet_mask'].bool(), 1, time_index]
        gf[graph.ndata['outlet_mask'].bool(), 1] = bcs[graph.ndata['outlet_mask'].bool(), 1, time_index]

    graph.ndata['nfeatures'] = gf

    delta = gnn_model(graph)
    delta[:,0] = nz.invert_normalize(delta[:,0], 'dp', params['statistics'],
                                     'labels')
    delta[:,1] = nz.invert_normalize(delta[:,1], 'dq', params['statistics'],
                                     'labels')
    gf[:,0:2] = gf[:,0:2] + delta

    return gf[:,0:2]

def rollout(gnn_model, params, graph):
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
        gf = perform_timestep(gnn_model, params, graph, tfc, it + 1)

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
           errs.detach().numpy(), end - start

    

    




