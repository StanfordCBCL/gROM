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

def compute_dataset_loss(gnn_model, train_dataloader, optimizer = None):
    global_loss = 0
    count = 0
    for batched_graph in train_dataloader:
        pred = gnn_model(batched_graph,
                         batched_graph.nodes['inner'].data['n_features'].float()).squeeze()
        weight = torch.ones(pred.shape)
        loss = weighted_mse_loss(pred,
                                 torch.reshape(batched_graph.nodes['inner'].data['n_labels'].float(),
                                 pred.shape), weight)
        time = batched_graph.nodes['inlet'].data['time'].detach().numpy()
        global_loss = global_loss + loss.detach().numpy()
        if optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        count = count + 1

    return global_loss, count

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
        global_loss, count = compute_dataset_loss(gnn_model, train_dataloader, optimizer)
        scheduler.step()
        print('\tloss = ' + str(global_loss / count))

        if checkpoint_fct != None:
            if epoch in chckp_epochs:
                checkpoint_fct(global_loss/count)

    # compute final loss
    global_loss, count = compute_dataset_loss(gnn_model, train_dataloader)
    print('\tFinal loss = ' + str(global_loss / count))

    return gnn_model, train_dataloader, global_loss / count, coefs_dict, dataset

def evaluate_error(model, model_name, train_dataloader, coefs_dict, do_plot, out_folder):
    it = iter(train_dataloader)
    batch = next(it)
    batched_graph = batch
    graph = dgl.unbatch(batched_graph)[0]

    true_graph = load_graphs('../graphs/data/' + model_name + '.grph')[0][0]
    times = pp.get_times(true_graph)
    # times = times[0:200]
    new_pressure_inlet = pp.normalize_function(true_graph.nodes['inner'].data['pressure_' + str(times[0])],
                                               'pressure', coefs_dict)
    new_flowrate = pp.normalize_function(true_graph.nodes['inner'].data['flowrate_' + str(times[0])],
                                             'flowrate', coefs_dict)

    pressure_dict = {'inner': pp.normalize_function(true_graph.nodes['inner'].data['pressure_' + str(times[0])],
                             'pressure', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_' + str(times[0])],
                             'pressure', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_' + str(times[0])],
                             'pressure', coefs_dict)}
    flowrate_dict = {'inner': pp.normalize_function(true_graph.nodes['inner'].data['flowrate_' + str(times[0])],
                             'flowrate', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_' + str(times[0])],
                             'flowrate', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_' + str(times[0])],
                             'flowrate', coefs_dict)}

    new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

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

        next_pressure = true_graph.nodes['inner'].data['pressure_' + str(tp1)]
        next_flowrate = true_graph.nodes['inner'].data['flowrate_' + str(tp1)]

        pressure_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_' + str(tp1)],
                                 'pressure', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_' + str(tp1)],
                                 'pressure', coefs_dict)}
        flowrate_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)],
                                 'flowrate', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_' + str(tp1)],
                                 'flowrate', coefs_dict)}

        new_bcs = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

        pp.set_bcs(graph, new_bcs)
        pp.set_state(graph, new_state)
        pred = model(graph, graph.nodes['inner'].data['n_features'].float()).squeeze()

        if (1):
            dp = pp.invert_normalize_function(pred[:,0].detach().numpy(), 'dp', coefs_dict)
            prev_p = pp.invert_normalize_function(graph.nodes['inner'].data['pressure'].detach().numpy().squeeze(),
                                                  'pressure', coefs_dict)

            p = dp + prev_p
            # print(np.linalg.norm(p))
            pressures_pred.append(p)
            pressures_real.append(next_pressure.detach().numpy())

            dq = pp.invert_normalize_function(pred[:,1].detach().numpy(), 'dq', coefs_dict)
            prev_q = pp.invert_normalize_function(graph.nodes['inner'].data['flowrate'].detach().numpy().squeeze(),
                                                  'flowrate', coefs_dict)

            q = dq + prev_q

            flowrates_pred.append(q)
            flowrates_real.append(next_flowrate.detach().numpy())
        else:
            p = pp.invert_normalize_function(pred[:,0].detach().numpy(), 'pressure', coefs_dict)

            # print(np.linalg.norm(p))
            pressures_pred.append(p)
            pressures_real.append(next_pressure.detach().numpy())

            q = pp.invert_normalize_function(pred[:,1].detach().numpy(), 'flowrate', coefs_dict)

            flowrates_pred.append(q)
            flowrates_real.append(next_flowrate.detach().numpy())

        err_p = err_p + np.linalg.norm(p - next_pressure.detach().numpy().squeeze())**2
        norm_p = norm_p + np.linalg.norm(next_pressure.detach().numpy().squeeze())**2
        err_q = err_q + np.linalg.norm(q - next_flowrate.detach().numpy().squeeze())**2
        norm_q = norm_q + np.linalg.norm(next_flowrate.detach().numpy().squeeze())**2

        pressure_dict = {'inner': torch.from_numpy(np.expand_dims(pp.normalize_function(p,'pressure', coefs_dict),axis=1)),
                         'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_' + str(tp1)],
                                 'pressure', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_' + str(tp1)],
                                 'pressure', coefs_dict)}
        flowrate_dict = {'inner': torch.from_numpy(np.expand_dims(pp.normalize_function(p,'flowrate', coefs_dict),axis=1)),
                         'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)],
                                 'flowrate', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_' + str(tp1)],
                                 'flowrate', coefs_dict)}

        new_state = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

    err_p = np.sqrt(err_p / norm_p)
    err_q = np.sqrt(err_q / norm_q)

    if do_plot:
        pressures_pred = pressures_pred[0::10]
        pressures_real = pressures_real[0::10]
        flowrates_pred = flowrates_pred[0::10]
        flowrates_real = flowrates_real[0::10]

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
            ax[0].set_ylim(coefs_dict['pressure']['min']-np.abs(coefs_dict['pressure']['min'])*0.1,coefs_dict['pressure']['max']+np.abs(coefs_dict['pressure']['max'])*0.1)
            ax[1].set_xlim(0,len(flowrates_pred[i]))
            ax[1].set_ylim(coefs_dict['flowrate']['min']-np.abs(coefs_dict['flowrate']['min'])*0.1,coefs_dict['flowrate']['max']+np.abs(coefs_dict['flowrate']['max'])*0.1)
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
    dataset.save_graphs(folder)
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
                    'nepochs': 1}
    dataset_params = {'rate_noise': 1e-1,
                      'random_walks': 4,
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
