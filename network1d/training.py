import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../graph1d")
sys.path.append("../tools")

import argparse
import time
import io_utils as io
import numpy as np
import torch.distributed as dist
import generate_dataset as dset
import torch as th
from datetime import datetime
from meshgraphnet import MeshGraphNet
import pathlib
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

def mse(input, target):
    return ((input - target) ** 2).mean()

def mae(input, target, weight = None):
    if weight == None:
        return (th.abs(input - target)).mean()
    return (weight * (th.abs(input - target))).mean()

def evaluate_model(gnn_model, train_dataloader, test_dataloader, optimizer):
    def loop_over(dataloader, label, c_optimizer = None):
        global_loss = 0
        global_metric = 0
        count = 0

        for batched_graph in tqdm(dataloader, 
                                  desc = label, colour='green'):
            pred = gnn_model(batched_graph)

            loss_v = mse(pred, batched_graph.ndata['nlabels'])
            global_loss = global_loss + loss_v.detach().numpy()

            metric_v = mae(pred, batched_graph.ndata['nlabels'])
            global_metric = global_metric + metric_v.detach().numpy()

            if c_optimizer != None:
                optimizer.zero_grad()
                loss_v.backward()
                optimizer.step()
            
            count = count + 1

        return {'loss': global_loss / count, 'metric': global_metric / count}

    start = time.time()
    test_results = loop_over(test_dataloader, 'test')
    train_results = loop_over(train_dataloader, 'train', optimizer)
    end = time.time()

    return train_results, test_results, end - start

def train_gnn_model(gnn_model, dataset, params, parallel, 
                    checkpoint_fct = None):
    if parallel:
        train_sampler = DistributedSampler(dataset['train'], 
                                           num_replicas = dist.get_world_size(),
                                           rank = dist.get_rank())
        test_sampler = DistributedSampler(dataset['test'],      
                                          num_replicas=dist.get_world_size(), 
                                          rank=dist.get_rank())
    else: 
        num_train = int(len(dataset['train']))
        train_sampler = SubsetRandomSampler(th.arange(num_train))
        num_test = int(len(dataset['test']))
        test_sampler = SubsetRandomSampler(th.arange(num_test))
    
    train_dataloader = GraphDataLoader(dataset['train'], 
                                       sampler = train_sampler,
                                       batch_size = params['batch_size'],
                                       drop_last = False)

    test_dataloader = GraphDataLoader(dataset['test'], 
                                      sampler = test_sampler,
                                      batch_size = params['batch_size'],
                                      drop_last = False)

    if parallel:
        print("my rank = %d, world = %d, train_dataloader_len = %d." \
        % (dist.get_rank(), dist.get_world_size(), len(train_dataloader)),\
        flush=True)
    
    optimizer = th.optim.Adam(gnn_model.parameters(),
                              params['learning_rate'],
                              weight_decay= params['weight_decay'])

    nepochs = params['nepochs']

    eta_min = params['learning_rate'] * params['lr_decay']
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max = nepochs,
                                                        eta_min = eta_min)

    for epoch in range(nepochs):
        dataset['train'].set_noise_rate(params['rate_noise'])
        dataset['test'].set_noise_rate(params['rate_noise'])

        train_results, test_results, elapsed = evaluate_model(gnn_model,
                                                              train_dataloader,
                                                              test_dataloader,
                                                              optimizer)

        msg = '{:.0f}\t'.format(epoch)
        msg = msg + 'train_loss = {:.2e} '.format(train_results['loss'])
        msg = msg + 'train_mae = {:.2e} '.format(train_results['metric'])
        msg = msg + 'test_loss = {:.2e} '.format(test_results['loss'])
        msg = msg + 'test_mae = {:.2e} '.format(test_results['metric'])
        msg = msg + 'time = {:.2f} s'.format(elapsed)

        print(msg, flush=True)

        scheduler.step()

def launch_training(dataset, params, parallel, out_dir = 'models/', 
                    checkpoint_fct = None):
    now = datetime.now()
    folder = out_dir + now.strftime("%d.%m.%Y_%H.%M.%S")

    gnn_model = MeshGraphNet(params)
    def save_model(filename):
        if parallel:
            th.save(gnn_model.module.state_dict(), folder + '/' + filename)
        else:
            th.save(gnn_model.state_dict(),  folder + '/' + filename)

    def default(obj):
        if isinstance(obj, th.Tensor):
            return default(obj.detach().numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        print(obj)
        raise TypeError('Not serializable')

    save_data = True
    if parallel:
        gnn_model = th.nn.parallel.DistributedDataParallel(gnn_model)
        save_data = (dist.get_rank() == 0)

    if save_data:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        save_model('initial_gnn.pms')

    train_gnn_model(gnn_model, dataset, params, checkpoint_fct)

if __name__ == "__main__":
    try:
        parallel = True
        dist.init_process_group(backend='mpi')
        print("my rank = %d, world = %d." % (dist.get_rank(), dist.get_world_size()), flush=True)
    except RuntimeError:
        parallel = False
        print("MPI not supported. Running serially.")

    data_location = io.data_location()
    input_dir = data_location + 'graphs/'
    output_dir = data_location + 'normalized_graphs/'

    parser = argparse.ArgumentParser(description='Graph Reduced Order Models')

    parser.add_argument('--bs', help='batch size', type=int, default=200)
    parser.add_argument('--epochs', help='total number of epochs', type=int, default=1000)
    parser.add_argument('--lr_decay', help='learning rate decay', type=float, default=0.1)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.008)
    parser.add_argument('--rate_noise', help='rate noise', type=float, default=1)
    parser.add_argument('--continuity_coeff', help='continuity coefficient', type=int, default=-3)
    parser.add_argument('--bc_coeff', help='boundary conditions coefficient', type=int, default=-5)
    parser.add_argument('--momentum', help='momentum', type=float, default=0)
    parser.add_argument('--weight_decay', help='weight decay for l2 regularization', type=float, default=1e-5)
    parser.add_argument('--ls_gnn', help='latent size gnn', type=int, default=16)
    parser.add_argument('--ls_mlp', help='latent size mlps', type=int, default=64)
    parser.add_argument('--process_iterations', help='gnn layers', type=int, default=2)
    parser.add_argument('--hl_mlp', help='hidden layers mlps', type=int, default=1)
    parser.add_argument('--nmc', help='copies per model', type=int, default=1)

    args = parser.parse_args()

    params = {'infeat_edges': 4,
              'latent_size_gnn': args.ls_gnn,
              'latent_size_mlp': args.ls_mlp,
              'out_size': 2,
              'process_iterations': args.process_iterations,
              'number_hidden_layers_mlp': args.hl_mlp,
              'learning_rate': args.lr,
              'momentum': args.momentum,
              'batch_size': args.bs,
              'lr_decay': args.lr_decay,
              'nepochs': args.epochs,
              'continuity_coeff': args.continuity_coeff,
              'bc_coeff': args.bc_coeff,
              'weight_decay': args.weight_decay,
              'rate_noise': args.rate_noise}
    start = time.time()

    data_location = io.data_location()
    input_dir = data_location + 'normalized_graphs/'
    datasets = dset.generate_dataset(input_dir)

    for dataset in datasets:
        launch_training(dataset, params, parallel)

    end = time.time()
    elapsed_time = end - start
    print('Training time = ' + str(elapsed_time))