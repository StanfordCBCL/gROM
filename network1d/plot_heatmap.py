import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import graph1d.generate_normalized_graphs as gng
import matplotlib
from matplotlib import animation
import torch as th
from typing import Any
import os
import plotly.graph_objects as go
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from random import sample
import graph1d.generate_normalized_graphs as nz
import matplotlib.cm as cm
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import tools.io_utils as io
import json
from meshgraphnet import MeshGraphNet
from rollout import rollout

def reorder_features(features, graph):
    branch_nodes = np.where(graph.ndata['type'][:,1,:] != 1)[0]
    juncti_nodes = np.where(graph.ndata['type'][:,1,:] == 1)[0]
    print(branch_nodes)
    print(juncti_nodes)
    return np.concatenate((features[branch_nodes,:],features[juncti_nodes,:],), 
           axis = 0), len(branch_nodes)

if __name__ == "__main__":
    data_location = io.data_location()
    input_dir = data_location + 'graphs/'
    model_dir = sys.argv[1]
    graph_name = sys.argv[2]
    params = json.load(open(model_dir + '/parameters.json'))

    graphs, _ = gng.generate_normalized_graphs(input_dir, 
                                    params['statistics']['normalization_type'],
                                    params['bc_type'])

    
    gnn_model = MeshGraphNet(params)
    gnn_model.load_state_dict(th.load(model_dir + '/trained_gnn.pms'))
    reference_graph = graphs[graph_name]
    r_features, _, _ = rollout(gnn_model, params, reference_graph)
    r_features, n_branch = reorder_features(r_features, reference_graph)
    r_features[:,0,:] = gng.invert_normalize(r_features[:,0,:], 'pressure', 
                         params['statistics'],'features')
    reference_features = reference_graph.ndata['nfeatures'].detach().numpy()
    reference_features, _ = reorder_features(reference_features,
                                             reference_graph)
    reference_features = gng.invert_normalize(reference_features, 'pressure', 
                         params['statistics'],'features')

    fig = plt.figure(figsize =(7, 5))
    ax = fig.add_subplot(121)
    nnodes = r_features.shape[0]
    ntimes = r_features.shape[2]
    dt = float(gng.invert_normalize(reference_graph.ndata['dt'][0], 
                              'dt', params['statistics'], 'features'))
    print(nnodes)
    print(dt)
    print(ntimes)
    pos = ax.imshow(r_features[::-1,0,:], cmap='viridis', 
              interpolation='nearest', extent=[0 , ntimes * dt, 0, nnodes],
              vmin = np.min(reference_features[:,0,:]),
              vmax = np.max(reference_features[:,0,:]),
              aspect='auto')
    print(r_features[:,0,:].shape)
    ax.set_ylabel('Node index')
    ax.set_xlabel('Time [s]')
    ax.plot([0, ntimes * dt], [n_branch, n_branch], '--',
            linewidth = 2, color = 'red')
    ax.set_title('GNN')

    ax = fig.add_subplot(122)
    pos = ax.imshow(reference_features[::-1,0,:], 
              cmap='viridis', interpolation='nearest',
              extent=[0, ntimes * dt, 0, nnodes],
              aspect='auto')
    ax.set_xlabel('Time [s]')
    ax.axes.yaxis.set_ticks([])
    ax.set_title('Ground truth')

    cb_ax = fig.add_axes([.91,.11,.02,.77])
    fig.colorbar(pos, orientation='vertical',cax=cb_ax)

    ax.plot([0, ntimes * dt], [n_branch, n_branch], '--',
            linewidth = 2, color = 'red')

    plt.show()

    
