import sys
import os
sys.path.append(os.getcwd())
import torch as th
import graph1d.generate_normalized_graphs as gng
import graph1d.generate_dataset as dset
import tools.io_utils as io
from network1d.meshgraphnet import MeshGraphNet
import json
import shutil
import pathlib
from network1d.rollout import rollout
import tools.plot_tools as pt

def plot_rollout(features, graph, params, folder, filename = 'all_nodes.mp4'):
    """
    Saves videos with all nodal values for pressure and flow rate, for all
    timesteps

    Arguments:
        features: 3D array containing the GNN prediction. 
                  1 dim: graph nodes, 2 dim: pressure (0), florate (1),
                  3 dim: timestep index
        params: dictionary of parameters
        folder (string): path where the video should be saved
        file_name: name of output file. Default -> 'all_nodes.mp4'

    """
    pt.video_all_nodes(features, graph, params, 5, folder + filename)

def evaluate_all_models(dataset, split_name, gnn_model, params, doplot = False):
    """
    Runs the rollout phase for all models and computes errors.

    Arguments:
        dataset: dictionary containing two keys, 'train' and 'test', with
                 the different datasets
        split_name (string): either 'train' or 'test'
        gnn_model: the GNN
        params: dictionary of parameters
        doplot: if True, the functions creates and saves one video per simulation. Default -> False
    Returns:
        2D array containing average pressure and flow rate normalized errors
        2D array containing average pressure and flow rate errors, 
        Average continuity loss
        Average run time
        Average number of timesteps

    """
    print('==========' + split_name + '==========')
    dataset = dataset[split_name]
    if doplot:
        pathlib.Path('results/' + split_name).mkdir(parents=True, exist_ok=True)

    total_timesteps = 0
    total_time = 0
    tot_errs_normalized = 0
    tot_errs = 0
    tot_cont_loss = 0
    for i in range(0,len(dataset.graphs)):
        print('model name = {}'.format(dataset.graph_names[i]))
        fdr = 'results/' + split_name + '/' + dataset.graph_names[i] + '/'
        pathlib.Path(fdr).mkdir(parents=True, exist_ok=True)
        r_features, errs_normalized, \
        errs,_,cont_loss, elaps = rollout(gnn_model, params, dataset.graphs[i])
        total_time = total_time + elaps
        total_timesteps = total_timesteps + r_features.shape[2]
        print('Errors normalized')
        print(errs_normalized)
        print('Errors')
        print(errs)
        print('Continuity loss')
        print(cont_loss)
        if doplot:
            plot_rollout(r_features, dataset.graphs[i], params, fdr)
        tot_errs_normalized = tot_errs_normalized + errs_normalized
        tot_errs = tot_errs + errs
        tot_cont_loss = tot_cont_loss + cont_loss

    N = len(dataset.graphs)
    print('-------------------------------------')
    print('Global statistics')
    print('Errors normalized')
    print(tot_errs_normalized / N)
    print('Errors')
    print(tot_errs / N)
    print('Continuity loss')
    print(tot_cont_loss / N)
    print('Average time = {:.2f}'.format(total_time / N))
    print('Average n timesteps = {:.2f}'.format(total_timesteps / N))

    return tot_errs_normalized/N, tot_errs/N, tot_cont_loss/N, \
           total_time / N, total_timesteps / N

def get_dataset_and_gnn(path, data_location = None):
    """
    Get datasets and GNN given the path to a saved model folder.

    Arguments:
        path (string): path to the GNN model folder. This should be the output
                       generated when launching the 'network1d/training.py'
                       script
        data_location (string): location of the 'gROM_data' folder. If None, 
                                we take the default location (which must be 
                                specified in data_location.txt).
                                Default -> None
    Returns:
        Dictionary containing train and test datasets
        GNN model
        Dictionary containing parameters

    """
    params = json.load(open(path + '/parameters.json'))

    gnn_model = MeshGraphNet(params)
    gnn_model.load_state_dict(th.load(path + '/trained_gnn.pms'))

    if data_location == None:
        data_location = io.data_location()
    graphs, _  = gng.generate_normalized_graphs(data_location + 'graphs/',
                                                params['statistics']['normalization_type'],
                                                params['bc_type'],
                                                statistics = params['statistics'])


    dataset = dset.generate_dataset_from_params(graphs, params)
    return dataset, gnn_model, params

"""
This function expects the location of a saved model folder as command line
argument. This is typically located in 'models/' after launching
'network1d/training.py'.
"""
if __name__ == '__main__':
    path = sys.argv[1]

    dataset, gnn_model, params = get_dataset_and_gnn(path)

    if os.path.exists('results'):
        shutil.rmtree('results')

    evaluate_all_models(dataset, 'train', gnn_model, params, True)
    evaluate_all_models(dataset, 'test', gnn_model, params, False)