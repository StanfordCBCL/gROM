import sys
import os
sys.path.append(os.getcwd())
import argparse
import time
import tools.io_utils as io
import numpy as np
import torch.distributed as dist
import graph1d.generate_dataset as dset
import torch as th
from datetime import datetime
from meshgraphnet import MeshGraphNet
import pathlib
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from rollout import rollout
import json
import tools.plot_tools as ptools
import pickle
import signal
import graph1d.generate_normalized_graphs as gng

class SignalHandler(object):
    def __init__(self):
        self.should_exit = False
    def handle(self, sig, frm):
        res = input("Do you want to exit training?" + \
                        "Model and statistics will be saved (y/n)")
        if res == "y":
            self.should_exit = True
        else:
            pass

def mse(input, target, mask = None):
    if mask == None:
        return ((input - target) ** 2).mean()
    return (mask * (input - target) ** 2).mean() 

def mae(input, target, mask = None):
    if mask == None:
        return (th.abs(input - target)).mean()
    return (mask * (th.abs(input - target))).mean()

def evaluate_model(gnn_model, train_dataloader, test_dataloader, optimizer,     
                   print_progress, params):
    def loop_over(dataloader, label, c_optimizer = None):
        global_loss = 0
        global_metric = 0
        count = 0

        def iteration(batched_graph, c_optimizer):
            pred = gnn_model(batched_graph)

            mask = th.ones(batched_graph.ndata['nlabels'].shape)
            if params['bc_type'] == 'realistic_dirichlet':
                mask[:,0] = mask[:,0] - batched_graph.ndata['outlet_mask']
                mask[:,1] = mask[:,1] - batched_graph.ndata['inlet_mask']
            elif params['bc_type'] == 'full_dirichlet':
                mask[:,0] = mask[:,0] - batched_graph.ndata['inlet_mask']
                mask[:,0] = mask[:,0] - batched_graph.ndata['outlet_mask']
                mask[:,1] = mask[:,1] - batched_graph.ndata['inlet_mask']
                mask[:,1] = mask[:,1] - batched_graph.ndata['outlet_mask']

            loss_v = mse(pred, batched_graph.ndata['nlabels'], mask)

            metric_v = mae(pred, batched_graph.ndata['nlabels'], mask)

            if c_optimizer != None:
                optimizer.zero_grad()
                loss_v.backward()
                optimizer.step()
            
            return loss_v.detach().numpy(), metric_v.detach().numpy()


        if not print_progress:
            for batched_graph in dataloader:
                loss_v, metric_v = iteration(batched_graph, c_optimizer)
                global_loss = global_loss + loss_v
                global_metric = global_metric + metric_v
                count = count + 1
        else:
            for batched_graph in tqdm(dataloader, 
                                    desc = label, colour='green'):
                loss_v, metric_v = iteration(batched_graph, c_optimizer)
                global_loss = global_loss + loss_v
                global_metric = global_metric + metric_v
                count = count + 1
                # total_norm = 0
                # for p in gnn_model.parameters():
                #     param_norm = p.grad.detach().data.norm(2)
                #     total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** 0.5
                # print(total_norm)

        return {'loss': global_loss / count, 'metric': global_metric / count}

    gnn_model.train()
    start = time.time()
    train_results = loop_over(train_dataloader, 'train', optimizer)
    test_results = loop_over(test_dataloader, 'test ')
    end = time.time()

    return train_results, test_results, end - start

def compute_rollout_errors(gnn_model, params, dataset, idxs_train, idxs_test):
    train_errs = np.zeros(2)
    for idx in idxs_train:
        _, cur_train_errs = rollout(gnn_model, params, dataset['train'],
                                    idx)
        train_errs = cur_train_errs + train_errs
    
    train_errs = train_errs / len(idxs_train)

    test_errs = np.zeros(2)
    for idx in idxs_test:
        _, cur_test_errs = rollout(gnn_model, params, dataset['test'],
                                    idx)
        test_errs = cur_test_errs + test_errs
    
    test_errs = test_errs / (len(idxs_test))

    return train_errs, test_errs

def train_gnn_model(gnn_model, dataset, params, parallel, rank0,
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
                              weight_decay = params['weight_decay'])

    nepochs = params['nepochs']

    eta_min = params['learning_rate'] * params['lr_decay']
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max = nepochs,
                                                        eta_min = eta_min)

    # sample train and test graphs for rollout
    np.random.seed(10)
    ngraphs = 2
    idxs_train = np.random.randint(0, len(dataset['train'].graphs), (ngraphs))
    idxs_test = np.random.randint(0, len(dataset['test'].graphs), (ngraphs))

    s = SignalHandler()
    history = {}
    history['train_loss'] = [[], []]
    history['train_metric'] = [[], []]
    history['train_rollout'] = [[], []]
    history['test_loss'] = [[], []]
    history['test_metric'] = [[], []]
    history['test_rollout'] = [[], []]         
    for epoch in range(nepochs):
        if rank0:
            print('================{}================'.format(epoch))
        dataset['train'].set_noise_rate(params['rate_noise'])
        dataset['test'].set_noise_rate(params['rate_noise'])

        train_results, test_results, elapsed = evaluate_model(gnn_model,
                                                              train_dataloader,
                                                              test_dataloader,
                                                              optimizer,
                                                              rank0,
                                                              params)

        msg = '{:.0f}\t'.format(epoch)
        msg = msg + 'train_loss = {:.2e} '.format(train_results['loss'])
        msg = msg + 'train_mae = {:.2e} '.format(train_results['metric'])
        msg = msg + 'test_loss = {:.2e} '.format(test_results['loss'])
        msg = msg + 'test_mae = {:.2e} '.format(test_results['metric'])
        msg = msg + 'time = {:.2f} s'.format(elapsed)

        if rank0:
            print("", flush = True)
            print(msg, flush = True)

        history['train_loss'][0].append(epoch)
        history['train_loss'][1].append(float(train_results['loss']))
        history['train_metric'][0].append(epoch)
        history['train_metric'][1].append(float(train_results['metric']))

        history['test_loss'][0].append(epoch)
        history['test_loss'][1].append(float(test_results['loss']))
        history['test_metric'][0].append(epoch)
        history['test_metric'][1].append(float(test_results['metric']))

        if epoch % np.floor(nepochs / 10) == 0 or epoch == (nepochs - 1):
            e_train, e_test = compute_rollout_errors(gnn_model, 
                                                     params, dataset, 
                                                     idxs_train, idxs_test)

            history['train_rollout'][0].append(epoch)
            history['train_rollout'][1].append(float(np.mean(e_train)))
            history['test_rollout'][0].append(epoch)
            history['test_rollout'][1].append(float(np.mean(e_test)))

        if rank0:
            msg = 'Rollout: {:.0f}\t'.format(epoch)
            print(msg, flush = True)
            print(history['train_rollout'][1])
            print(history['test_rollout'][1])

        scheduler.step()

        signal.signal(signal.SIGINT, s.handle)

        if s.should_exit:
            return gnn_model, history            

    return gnn_model, history

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

    gnn_model, history = train_gnn_model(gnn_model, dataset, params, 
                                         parallel, save_data, checkpoint_fct)

    if save_data:
        ptools.plot_history(history['train_loss'],
                        history['test_loss'],
                        'loss', folder)
        ptools.plot_history(history['train_metric'],
                        history['test_metric'],
                        'metric', folder)
        ptools.plot_history(history['train_rollout'],
                            history['test_rollout'],
                            'rollout', folder)
        save_model('trained_gnn.pms')

        with open(folder + '/history.bnr', 'wb') as outfile:
            pickle.dump(history, outfile)
        
        with open(folder + '/parameters.json', 'w') as outfile:
            json.dump(params, outfile, default=default, indent=4)

    return gnn_model

if __name__ == "__main__":
    rank = 0
    try:
        parallel = True
        dist.init_process_group(backend='mpi')
        rank = dist.get_rank()
        print("my rank = %d, world = %d." % (rank, dist.get_world_size()), flush=True)
    except RuntimeError:
        parallel = False
        print("MPI not supported. Running serially.")

    parser = argparse.ArgumentParser(description='Graph Reduced Order Models')

    parser.add_argument('--bs', help='batch size', type=int, default=100)
    parser.add_argument('--epochs', help='total number of epochs', type=int,
                        default=100)
    parser.add_argument('--lr_decay', help='learning rate decay', type=float,
                        default=0.1)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.005)
    parser.add_argument('--rate_noise', help='rate noise', type=float,
                        default=100)
    parser.add_argument('--weight_decay', help='l2 regularization', 
                        type=float, default=0)
    parser.add_argument('--ls_gnn', help='latent size gnn', type=int,
                        default=32)
    parser.add_argument('--ls_mlp', help='latent size mlps', type=int,
                        default=32)
    parser.add_argument('--process_iterations', help='gnn layers', type=int,
                        default=3)
    parser.add_argument('--hl_mlp', help='hidden layers mlps', type=int,
                        default=1)

    args = parser.parse_args()

    data_location = io.data_location()
    input_dir = data_location + 'graphs/'
    norm_type = {'features': 'normal', 'labels': 'min_max'}
    graphs, params  = gng.generate_normalized_graphs(input_dir, norm_type, 
                                                     'full_dirichlet')
    datasets = dset.generate_dataset(graphs, params)
    print(params)

    graph = graphs[list(graphs)[0]]

    infeat_nodes = graph.ndata['nfeatures'].shape[1]
    infeat_edges = graph.edata['efeatures'].shape[1]
    nout = graph.ndata['nlabels'].shape[1]

    t_params = {'infeat_nodes': infeat_nodes,
                'infeat_edges': infeat_edges,
                'latent_size_gnn': args.ls_gnn,
                'latent_size_mlp': args.ls_mlp,
                'out_size': nout,
                'process_iterations': args.process_iterations,
                'number_hidden_layers_mlp': args.hl_mlp,
                'learning_rate': args.lr,
                'batch_size': args.bs,
                'lr_decay': args.lr_decay,
                'nepochs': args.epochs,
                'weight_decay': args.weight_decay,
                'rate_noise': args.rate_noise}
    params.update(t_params)

    start = time.time()
    for dataset in datasets:
        params['train_split'] = dataset['train'].graph_names
        params['test_split'] = dataset['test'].graph_names
        gnn_model = launch_training(dataset, params, parallel)
    end = time.time()
    elapsed_time = end - start

    if rank == 0:
        print('Training time = ' + str(elapsed_time))