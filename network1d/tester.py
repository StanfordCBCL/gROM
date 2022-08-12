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
    dataset = dataset[split_name]
    pathlib.Path('results/' + split_name).mkdir(parents=True, exist_ok=True)

    total_timesteps = 0
    total_time = 0
    tot_errs_normalized = 0
    tot_errs = 0
    for i in range(len(dataset.graphs)):
        print('model name = {}'.format(dataset.graph_names[i]))
        fdr = 'results/' + split_name + '/' + dataset.graph_names[i] + '/'
        pathlib.Path(fdr).mkdir(parents=True, exist_ok=True)
        r_features, errs_normalized, \
        errs, elaps = rollout(gnn_model, params, dataset.graphs[i])
        total_time = total_time + elaps
        total_timesteps = total_timesteps + r_features.shape[2]
        print('Errors normalized')
        print_rollout_errors(errs_normalized)
        print('Errors')
        print_rollout_errors(errs)
        plot_rollout(r_features, dataset.graphs[i], params, fdr)
        tot_errs_normalized = tot_errs_normalized + errs_normalized
        tot_errs = tot_errs + errs

    print('-------------------------------------')
    print('Global statistics')
    N = len(dataset.graphs)
    print('Errors normalized')
    print(tot_errs_normalized / N)
    print('Errors')
    print(tot_errs / N)
    print('Average time = {:.2f}'.format(total_time / N))
    print('Average n timesteps = {:.2f}'.format(total_timesteps / N))

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
    
    # evaluate_all_models(datasets[0], 'train', gnn_model, params)
    evaluate_all_models(datasets[0], 'test', gnn_model, params)