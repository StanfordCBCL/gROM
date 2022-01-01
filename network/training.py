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

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def generate_gnn_model(params_dict):
    return GraphNet(params_dict)

def train_gnn_model(gnn_model, model_name, optimizer_name, train_params,
                    checkpoint_fct = None, dataset_params = None):
    dataset, coefs_dict = pp.generate_dataset(model_name,
                                              dataset_params)
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
        raise ValueError('Optimizer ' + optimizerizer_name + ' not implemented')

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
        global_loss = 0
        count = 0
        for batched_graph in train_dataloader:
            pred = gnn_model(batched_graph,
                             batched_graph.ndata['n_features'].float()).squeeze()
            weight = torch.ones(pred.shape)
            # mask out values corresponding to boundary conditions
            inlets = np.where(batched_graph.ndata['inlet_mask'].detach().numpy() == 1)[0]
            outlets = np.where(batched_graph.ndata['outlet_mask'].detach().numpy() == 1)[0]
            weight[inlets,:] = 100
            weight[outlets,:] = 100
            # weight[inlets,1] = 0
            # weight[outlets,0] = 0
            # weight[:,1] = 0
            # weight[:,0] = 0
            loss = weighted_mse_loss(pred,
                                     torch.reshape(batched_graph.ndata['n_labels'].float(),
                                     pred.shape), weight)
            global_loss = global_loss + loss.detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = count + 1
        scheduler.step()
        print(batched_graph.ndata['n_features'].float())
        print(pred)
        print('\tloss = ' + str(global_loss / count))

        if checkpoint_fct != None:
            if epoch in chckp_epochs:
                checkpoint_fct(global_loss/count)

    return gnn_model, train_dataloader, global_loss / count, coefs_dict

def evaluate_error(model, model_name, train_dataloader, coefs_dict, do_plot, out_folder):
    it = iter(train_dataloader)
    batch = next(it)
    batched_graph = batch
    graph = dgl.unbatch(batched_graph)[0]

    true_graph = load_graphs('../graphs/data/' + model_name + '.grph')[0][0]
    times = pp.get_times(true_graph)
    # times = times[0:]
    new_pressure = pp.normalize_function(true_graph.ndata['pressure_' + str(times[0])],
                                             'pressure', coefs_dict)
    new_flowrate = pp.normalize_function(true_graph.ndata['flowrate_' + str(times[0])],
                                             'flowrate', coefs_dict)

    err_p = 0
    err_q = 0
    norm_p = 0
    norm_q = 0
    pressures_pred = []
    pressures_real = []
    flowrates_pred = []
    flowrates_real = []
    for tind in range(len(times)-1):
        t = times[tind]
        tp1 = times[tind+1]

        next_pressure = true_graph.ndata['pressure_' + str(tp1)]
        next_flowrate = true_graph.ndata['flowrate_' + str(tp1)]
        np_normalized = pp.normalize_function(next_pressure, 'pressure', coefs_dict)
        nf_normalized = pp.normalize_function(next_flowrate, 'flowrate', coefs_dict)
        graph = pp.set_bcs(graph, np_normalized, nf_normalized)
        graph = pp.set_state(graph, new_pressure, new_flowrate)
        pred = model(graph, graph.ndata['n_features'].float()).squeeze()

        if (0):
            dp = pp.invert_normalize_function(pred[:,0].detach().numpy(), 'dp', coefs_dict)
            prev_p = pp.invert_normalize_function(graph.ndata['pressure'].detach().numpy().squeeze(),
                                                  'pressure', coefs_dict)
    
            p = dp + prev_p
            # print(np.linalg.norm(p))
            pressures_pred.append(p)
            pressures_real.append(next_pressure.detach().numpy())
    
            dq = pp.invert_normalize_function(pred[:,1].detach().numpy(), 'dq', coefs_dict)
            prev_q = pp.invert_normalize_function(graph.ndata['flowrate'].detach().numpy().squeeze(),
                                                  'flowrate', coefs_dict)
    
            q = dq + prev_q
    
            flowrates_pred.append(q)
            flowrates_real.append(next_flowrate.detach().numpy())
        else:
            dp = pp.invert_normalize_function(pred[:,0].detach().numpy(), 'pressure', coefs_dict)
    
            p = dp 
            # print(np.linalg.norm(p))
            pressures_pred.append(p)
            pressures_real.append(next_pressure.detach().numpy())
    
            dq = pp.invert_normalize_function(pred[:,1].detach().numpy(), 'flowrate', coefs_dict)
    
            q = dq
    
            flowrates_pred.append(q)
            flowrates_real.append(next_flowrate.detach().numpy())

        err_p = err_p + np.linalg.norm(p - next_pressure.detach().numpy().squeeze())**2
        norm_p = norm_p + np.linalg.norm(next_pressure.detach().numpy().squeeze())**2
        err_q = err_q + np.linalg.norm(q - next_flowrate.detach().numpy().squeeze())**2
        norm_q = norm_q + np.linalg.norm(next_flowrate.detach().numpy().squeeze())**2

        new_pressure = torch.unsqueeze(torch.from_numpy(pp.normalize_function(p, 'pressure', coefs_dict)),1)
        new_flowrate = torch.unsqueeze(torch.from_numpy(pp.normalize_function(q, 'flowrate', coefs_dict)),1)

    err_p = np.sqrt(err_p / norm_p)
    err_q = np.sqrt(err_q / norm_q)

    if do_plot:
        fig, ax = plt.subplots(2)
        line_pred_p, = ax[0].plot([],[],'r')
        line_real_p, = ax[0].plot([],[],'--b')
        line_pred_q, = ax[1].plot([],[],'r')
        line_real_q, = ax[1].plot([],[],'--b')

        def animation_frame(i):
            line_pred_p.set_xdata(range(0,len(pressures_pred[i])))
            line_pred_p.set_ydata(pressures_pred[i])
            line_real_p.set_xdata(range(0,len(pressures_pred[i])))
            line_real_p.set_ydata(pressures_real[i])
            line_pred_q.set_xdata(range(0,len(flowrates_pred[i])))
            line_pred_q.set_ydata(flowrates_pred[i])
            line_real_q.set_xdata(range(0,len(flowrates_pred[i])))
            line_real_q.set_ydata(flowrates_real[i])
            ax[0].set_xlim(0,len(pressures_pred[i]))
            ax[0].set_ylim(coefs_dict['pressure']['min'],coefs_dict['pressure']['max'])
            ax[1].set_xlim(0,len(flowrates_pred[i]))
            ax[1].set_ylim(coefs_dict['flowrate']['min'],coefs_dict['flowrate']['max'])
            return line_pred_p, line_real_p, line_pred_q, line_real_q

        anim = animation.FuncAnimation(fig, animation_frame,
                                       frames=len(pressures_pred),
                                       interval=20)
        writervideo = animation.FFMpegWriter(fps=60)
        anim.save(out_folder + '/plot.mp4', writer = writervideo)

    return err_p, err_q, np.sqrt(err_p**2 + err_q**2)

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        pass

def launch_training(model_name, optimizer_name, params_dict,
                    train_params, plot_validation = True, checkpoint_fct = None,
                    dataset_params = None):
    create_directory('models')
    gnn_model = generate_gnn_model(params_dict)
    gnn_model, train_loader, loss, coefs_dict = train_gnn_model(gnn_model,
                                                                model_name,
                                                                optimizer_name,
                                                                train_params,
                                                                checkpoint_fct,
                                                                dataset_params)

    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
    folder = 'models/' + dt_string
    create_directory(folder)
    torch.save(gnn_model.state_dict(), folder + '/gnn.pms')
    json_params = json.dumps(params_dict, indent = 4)
    json_train = json.dumps(train_params, indent = 4)
    with open(folder + '/hparams.json', 'w') as outfile:
        json.dump(json_params, outfile)
    with open(folder + '/train.json', 'w') as outfile:
        json.dump(json_train, outfile)
    return gnn_model, loss, train_loader, coefs_dict, folder

if __name__ == "__main__":
    params_dict = {'infeat_nodes': 6,
                   'infeat_edges': 4,
                   'latent_size_gnn': 4,
                   'latent_size_mlp': 4,
                   'out_size': 2,
                   'process_iterations': 10,
                   'hl_mlp': 2,
                   'normalize': True}
    train_params = {'learning_rate': 0.005,
                    'weight_decay': 0.999,
                    'momentum': 0.0,
                    'resample_freq_timesteps': -1,
                    'batch_size': 1,
                    'nepochs': 100}
    dataset_params = {'rate_noise': 1e-5,
                      'random_walks': 0,
                      'normalization': 'standard',
                      'resample_freq_timesteps': 1}

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

    err_p, err_q, global_err = evaluate_error(gnn_model, sys.argv[1],
                                              train_dataloader,
                                              coefs_dict,
                                              do_plot = True,
                                              out_folder = out_fdr)

    print('Error pressure = ' + str(err_p))
    print('Error flowrate = ' + str(err_q))
    print('Global error = ' + str(global_err))
