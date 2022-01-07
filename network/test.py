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

if __name__ == "__main__":
    dataset_params = {'rate_noise': 1e-5,
                      'random_walks': 0,
                      'normalization': 'standard',
                      'resample_freq_timesteps': 1}

    path = '/Users/luca/Desktop/06.01.2022_23.12.11/'
    params = json.loads(json.load(open(path + 'hparams.json')))

    gnn_model = GraphNet(params)
    gnn_model.load_state_dict(torch.load(path + 'gnn.pms'))
    gnn_model.eval()

    test_train(gnn_model, '0063_1001', dataset_params)
