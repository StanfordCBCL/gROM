import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import dgl
import torch
import preprocessing as pp
from graphnet import GraphNet
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from datetime import datetime
import time
import preprocessing as pp
import json
import training


def test_train(gnn_model, model_name, dataset_params):
    dataset, coefs_dict = pp.generate_dataset(model_name,
                                              dataset_params)

    num_examples = len(dataset)
    num_train = int(num_examples)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler,
                                       batch_size=1,
                                       drop_last=False)

    global_loss, count = training.compute_dataset_loss(gnn_model, train_dataloader)
    print('\tFinal loss = ' + str(global_loss / count))

    # count = 0
    # global_loss = 0
    # for batched_graph in train_dataloader:
    #     pred = gnn_model(batched_graph,
    #                      batched_graph.nodes['inner'].data['n_features'].float()).squeeze()
    #     weight = torch.ones(pred.shape)
    #     loss = training.weighted_mse_loss(pred,
    #                              torch.reshape(batched_graph.nodes['inner'].data['n_labels'].float(),
    #                              pred.shape), weight)
    #     time = batched_graph.nodes['inlet'].data['time'].detach().numpy()
    #     if np.abs(time - 0.0418) < 1e-3:
    #         print(pred[0:10,:])
    #         print(batched_graph.nodes['inner'].data['n_labels'].float()[0:10,:])
    #         print(batched_graph.nodes['inlet'].data['time'])
    #         print(loss)
    #     global_loss = global_loss + loss.detach().numpy()
    #     count = count + 1
    # print('\tloss = ' + str(global_loss / count))

if __name__ == "__main__":
    dataset_params = {'rate_noise': 1e-5,
                      'random_walks': 0,
                      'normalization': 'standard',
                      'resample_freq_timesteps': 1}

    path = '/Users/luca/Desktop/06.01.2022_00.39.59/'
    # path = 'models/06.01.2022_16.27.15/'
    params = json.loads(json.load(open(path + 'hparams.json')))

    gnn_model = GraphNet(params)
    gnn_model.load_state_dict(torch.load(path + 'gnn.pms'))
    # gnn_model.load_state_dict(torch.load('current_gnn.pms'))
    gnn_model.eval()

    test_train(gnn_model, '0063_1001', dataset_params)
    # torch.save(gnn_model.state_dict(), 'current_gnn.pms')

    # start = time.time()
    # gnn_model, _, train_dataloader, coefs_dict, out_fdr = launch_training(sys.argv[1],
    #                                                                       'adam',
    #                                                                        params_dict,
    #                                                                        train_params,
    #                                                                        checkpoint_fct = None,
    #                                                                        dataset_params = dataset_params)
    # end = time.time()
    # elapsed_time = end - start
    # print('Training time = ' + str(elapsed_time))
    #
    # err_p, err_q, global_err = evaluate_error(gnn_model, sys.argv[1],
    #                                           train_dataloader,
    #                                           coefs_dict,
    #                                           do_plot = True,
    #                                           out_folder = out_fdr)
    #
    # print('Error pressure = ' + str(err_p))
    # print('Error flowrate = ' + str(err_q))
    # print('Global error = ' + str(global_err))
