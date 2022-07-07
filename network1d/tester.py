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

def print_rollout_errors(errors):
    print(errors)

def plot_rollout(features, graph, params, folder):
    pt.video_all_nodes(features, graph, params, 5, folder + 'all_nodes.mp4')

def evaluate_all_models(dataset, split_name, gnn_model, params):
    print('==========' + split_name + '==========')
    dataset = dataset['train']
    pathlib.Path('results/' + split_name).mkdir(parents=True, exist_ok=True)

    tot_errs = 0
    for i in range(len(dataset.graphs)):
        print('model name = {}'.format(dataset.graph_names[i]))
        fdr = 'results/' + split_name + '/' + dataset.graph_names[i] + '/'
        pathlib.Path(fdr).mkdir(parents=True, exist_ok=True)
        r_features, errs = rollout(gnn_model, params, dataset, i)
        print_rollout_errors(errs)
        plot_rollout(r_features, dataset.graphs[i], params, fdr)
        tot_errs = tot_errs + errs

    print('-------------------------------------')
    print('Global statistics')
    print(errs)

if __name__ == '__main__':
    print(sys.argv)
    path = sys.argv[1]

    params = json.load(open(path + '/parameters.json'))

    gnn_model = MeshGraphNet(params)
    gnn_model.load_state_dict(th.load(path + '/trained_gnn.pms'))

    data_location = io.data_location()
    graphs, _  = gng.generate_normalized_graphs(data_location + 'graphs/', 
                                                params['statistics']['normalization_type'], 
                                                params['bc_type'])
    datasets = dset.generate_dataset(graphs, params)

    if os.path.exists('results'):
        shutil.rmtree('results')
    evaluate_all_models(datasets[0], 'train', gnn_model, params)