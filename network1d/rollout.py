import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../tools")
sys.path.append("../graphs")

import generate_normalized_graphs as nz
import numpy as np
import torch as th
import copy

def rollout(gnn_model, params, dataset, index_graph):
    gnn_model.eval()
    graph = copy.deepcopy(dataset.lightgraphs[index_graph])
    true_graph = dataset.graphs[index_graph]

    graph.ndata['nfeatures'] = true_graph.ndata['nfeatures'][:,:,0].clone()
    graph.edata['efeatures'] = true_graph.edata['efeatures'].squeeze().clone()

    times = dataset.times[index_graph]

    r_features = graph.ndata['nfeatures'][:,0:2].unsqueeze(axis = 2)
    for it in range(times-1):
        delta = gnn_model(graph)
        delta[:,0] = nz.invert_normalize(delta[:,0], 'dp', params['statistics'])
        delta[:,1] = nz.invert_normalize(delta[:,1], 'dq', params['statistics'])
        gf = graph.ndata['nfeatures'][:,0:2]
        gf = gf + delta
        # set boundary conditions
        pb = true_graph.ndata['nfeatures'][graph.ndata['outlet_mask'].bool(),
                                           0,it + 1]
        qb = true_graph.ndata['nfeatures'][graph.ndata['inlet_mask'].bool(),
                                           1,it + 1]

        gf[graph.ndata['outlet_mask'].bool(),0] = pb
        gf[graph.ndata['inlet_mask'].bool(),1] = qb

        graph.ndata['nfeatures'][:,0:2] = gf.clone()
        r_features = th.cat((r_features, gf.unsqueeze(axis = 2)), axis = 2)

    tfc = true_graph.ndata['nfeatures'][:,0:2,:].clone()
    tfc[:,0] = nz.invert_normalize(tfc[:,0], 'pressure', params['statistics'])
    tfc[:,1] = nz.invert_normalize(tfc[:,1], 'flowrate', params['statistics'])

    rfc = r_features.clone()
    rfc[:,0] = nz.invert_normalize(rfc[:,0], 'pressure', params['statistics'])
    rfc[:,1] = nz.invert_normalize(rfc[:,1], 'flowrate', params['statistics'])

    # compute error
    diff = tfc - rfc
    errs = th.sum(th.sum(diff**2, dim = 0), dim = 1)
    errs = errs / th.sum(th.sum(tfc**2, dim = 0), dim = 1)
    errs = th.sqrt(errs)

    return r_features, errs
    

    




