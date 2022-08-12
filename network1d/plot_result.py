import sys
import os
sys.path.append(os.getcwd())
import torch as th
import graph1d.generate_normalized_graphs as gng
import graph1d.generate_dataset as dset
import tools.io_utils as io
from meshgraphnet import MeshGraphNet
import json
import shutil
import pathlib
from rollout import rollout
import tools.plot_tools as pt
import pickle

if __name__ == '__main__':
    path = sys.argv[1]

    params = json.load(open(path + '/parameters.json'))

    gnn_model = MeshGraphNet(params)
    gnn_model.load_state_dict(th.load(path + '/trained_gnn.pms'))

    with open(path + '/history.bnr', 'rb') as f:
        history = pickle.load(f)

    print('Train rollout')
    print(history['train_rollout'][1][-1])
    print('Test rollout')
    print(history['test_rollout'][1][-1])


    data_location = io.data_location()
    graphs, _  = gng.generate_normalized_graphs(data_location + 'graphs/', 
                                                params['statistics']['normalization_type'], 
                                                params['bc_type'])
    
    model_name = sys.argv[2]
    features, _, _ = rollout(gnn_model, params, graphs[model_name])
    real_features = graphs[model_name].ndata['nfeatures']

    pt.plot_curves(features, real_features, graphs[model_name].ndata['type'],
                   graphs[model_name].ndata['x'], 5, params, '.')
    pt.plot_history(history['train_loss'], history['test_loss'], 'loss', '.')
    pt.plot_history(history['train_rollout'], history['test_rollout'], 'relative rollout error', '.')

