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
import json

def mse(input, target):
    return ((input - target) ** 2).mean()

def mae(input, target):
    return (torch.abs(input - target)).mean()

def generate_gnn_model(params_dict):
    return GraphNet(params_dict)

def evaluate_model(gnn_model, train_dataloader, loss, metric = None, optimizer = None):
    global_loss = 0
    global_metric = 0
    count = 0
    start = time.time()
    for batched_graph in train_dataloader:
        pred = gnn_model(batched_graph,
                         batched_graph.nodes['inner'].data['n_features'].float()).squeeze()

        loss_v = loss(pred, torch.reshape(batched_graph.nodes['inner'].data['n_labels'].float(),
                      pred.shape))

        global_loss = global_loss + loss_v.detach().numpy()

        if metric != None:
            metric_v = metric(pred, torch.reshape(batched_graph.nodes['inner'].data['n_labels'].float(),
                              pred.shape))

            global_metric = global_metric + metric_v.detach().numpy()

        if optimizer != None:
            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()
        count = count + 1

    end = time.time()

    return global_loss, count, end - start, global_metric

def train_gnn_model(gnn_model, model_name, optimizer_name, train_params,
                    checkpoint_fct = None, dataset_params = None):
    dataset, coefs_dict = pp.generate_dataset(model_name,
                                              dataset_params)
    gnn_model.set_normalization_coefs(coefs_dict)
    num_examples = len(dataset)
    num_train = int(num_examples)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler,
                                       batch_size=train_params['batch_size'],
                                       drop_last=False)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     train_params['learning_rate'])
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(gnn_model.parameters(),
                                    train_params['learning_rate'],
                                    momentum=train_params['momentum'])
    else:
        raise ValueError('Optimizer ' + optimizer_name + ' not implemented')

    nepochs = train_params['nepochs']
    scheduler_name = 'cosine'
    if scheduler_name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=train_params['weight_decay'])
    elif scheduler_name == 'cosine':
        eta_min = train_params['learning_rate'] * train_params['weight_decay']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=nepochs,
                                                               eta_min=eta_min)

    if checkpoint_fct != None:
        # 200 is the maximum number of sigopt checkpoint
        chckp_epochs = list(np.floor(np.linspace(0, nepochs, 200)))

    for epoch in range(nepochs):
        print('ep = ' + str(epoch))
        global_loss, count, elapsed, global_mae = evaluate_model(gnn_model, train_dataloader, mse, mae, optimizer)
        scheduler.step()
        print('\tloss = {:.2e}\tmae = {:.2e}\ttime = {:.2f} s'.format(global_loss/count,
                                                                      global_mae/count,
                                                                      elapsed))

        if checkpoint_fct != None:
            if epoch in chckp_epochs:
                checkpoint_fct(global_loss/count)

    # compute final loss
    global_loss, count, _, global_mae = evaluate_model(gnn_model, train_dataloader, mse, mae)
    print('\tFinal loss = {:.2e}\tfinal mae = {:.2e}'.format(global_loss/count,
                                                             global_mae/count))

    return gnn_model, train_dataloader, global_loss / count, coefs_dict, dataset

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        pass

def launch_training(model_name, optimizer_name, params_dict,
                    train_params, plot_validation = True, checkpoint_fct = None,
                    dataset_params = None):
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
    folder = 'models/' + dt_string
    create_directory('models')
    create_directory(folder)
    gnn_model = generate_gnn_model(params_dict)
    torch.save(gnn_model.state_dict(), folder + '/initial_gnn.pms')
    gnn_model, train_loader, loss, coefs_dict, dataset = train_gnn_model(gnn_model,
                                                                         model_name,
                                                                         optimizer_name,
                                                                         train_params,
                                                                         checkpoint_fct,
                                                                         dataset_params)

    torch.save(gnn_model.state_dict(), folder + '/trained_gnn.pms')
    json_params = json.dumps(params_dict, indent = 4)
    json_train = json.dumps(train_params, indent = 4)
    with open(folder + '/hparams.json', 'w') as outfile:
        json.dump(json_params, outfile)
    with open(folder + '/train.json', 'w') as outfile:
        json.dump(json_train, outfile)
    return gnn_model, loss, train_loader, coefs_dict, folder

if __name__ == "__main__":
    params_dict = {'infeat_nodes': 7,
                   'infeat_edges': 4,
                   'latent_size_gnn': 18,
                   'latent_size_mlp': 84,
                   'out_size': 2,
                   'process_iterations': 5,
                   'hl_mlp': 1,
                   'normalize': 1}
    train_params = {'learning_rate': 0.008223127794360673,
                    'weight_decay': 0.36984122162067234,
                    'momentum': 0.0,
                    'batch_size': 359,
                    'nepochs': 10}
    dataset_params = {'normalization': 'standard'}

    start = time.time()
    gnn_model, _, train_dataloader, coefs_dict, out_fdr = launch_training(sys.argv[1],
                                                                          'adam',
                                                                           params_dict,
                                                                           train_params,
                                                                           checkpoint_fct = None,
                                                                           dataset_params = dataset_params)
    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))
