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

def test_train(gnn_model, model_name, dataset):
    num_examples = len(dataset)
    num_train = int(num_examples)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler,
                                       batch_size=1,
                                       drop_last=False)

    global_loss, count, elapsed = training.compute_dataset_loss(gnn_model, train_dataloader)
    print('\tFinal loss = ' + str(global_loss / count))
    print('\t\tcomputed in ' + str(elapsed) + 's')

    return coefs_dict

def test_rollout(model, model_name, graph, coefs_dict, do_plot, out_folder):
    true_graph = load_graphs('../graphs/data/' + model_name + '.grph')[0][0]
    times = pp.get_times(true_graph)

    pressure_dict = {'inner': pp.normalize_function(true_graph.nodes['inner'].data['pressure_0'],
                             'pressure', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_0'],
                             'pressure', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_0'],
                             'pressure', coefs_dict)}
    flowrate_dict = {'inner': pp.normalize_function(true_graph.nodes['inner'].data['flowrate_0'],
                             'flowrate', coefs_dict),
                     'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_0'],
                             'flowrate', coefs_dict),
                     'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_0'],
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
    for t in range(len(times)-1):
        if (t % 100 == 0 or t == (len(times)-2)):
            print('Rollout ' + str(t) + '/' + str(len(times)-1))
        tp1 = t+1

        next_pressure = pp.normalize_function(true_graph.nodes['inner'].data['pressure_' + str(tp1)], 'pressure', coefs_dict)
        next_flowrate = pp.normalize_function(true_graph.nodes['inner'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict)

        pressure_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['pressure_' + str(tp1)], 'pressure', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['pressure_' + str(tp1)], 'pressure', coefs_dict)}
        flowrate_dict = {'inlet': pp.normalize_function(true_graph.nodes['inlet'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict),
                         'outlet': pp.normalize_function(true_graph.nodes['outlet'].data['flowrate_' + str(tp1)], 'flowrate', coefs_dict)}

        new_bcs = {'pressure': pressure_dict, 'flowrate': flowrate_dict}

        pp.set_bcs(graph, new_bcs)
        pp.set_state(graph, new_state)
        pred = model(graph, graph.nodes['inner'].data['n_features'].float()).squeeze()

        dp = pred[:,0].detach().numpy()
        prev_p = graph.nodes['inner'].data['pressure'].detach().numpy().squeeze()

        p = dp + prev_p

        pressures_pred.append(p)
        pressures_real.append(next_pressure.detach().numpy())

        dq = pred[:,1].detach().numpy()
        prev_q = graph.nodes['inner'].data['flowrate'].detach().numpy().squeeze()

        q = dq + prev_q

        flowrates_pred.append(q)
        flowrates_real.append(next_flowrate.detach().numpy())

        err_p = err_p + np.linalg.norm(p - next_pressure.detach().numpy().squeeze())**2
        norm_p = norm_p + np.linalg.norm(next_pressure.detach().numpy().squeeze())**2
        err_q = err_q + np.linalg.norm(q - next_flowrate.detach().numpy().squeeze())**2
        norm_q = norm_q + np.linalg.norm(next_flowrate.detach().numpy().squeeze())**2

        pressure_dict = {'inner': torch.from_numpy(np.expand_dims(p,axis=1)),
                         'inlet': graph.nodes['inlet'].data['pressure_next'],
                         'outlet': graph.nodes['outlet'].data['pressure_next'],}
        flowrate_dict = {'inner': torch.from_numpy(np.expand_dims(q,axis=1)),
                         'inlet': graph.nodes['inlet'].data['flowrate_next'],
                         'outlet': graph.nodes['outlet'].data['flowrate_next']}

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
            line_pred_p.set_ydata(pp.invert_normalize_function(pressures_pred[i],'pressure',coefs_dict))
            line_real_p.set_xdata(range(0,len(pressures_pred[i])))
            line_real_p.set_ydata(pp.invert_normalize_function(pressures_real[i],'pressure',coefs_dict))
            line_pred_q.set_xdata(range(0,len(flowrates_pred[i])))
            line_pred_q.set_ydata(pp.invert_normalize_function(flowrates_pred[i],'flowrate',coefs_dict))
            line_real_q.set_xdata(range(0,len(flowrates_pred[i])))
            line_real_q.set_ydata(pp.invert_normalize_function(flowrates_real[i],'flowrate',coefs_dict))
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

if __name__ == "__main__":
    dataset_params = {'normalization': 'standard'}

    path = 'models/08.01.2022_17.26.24/'
    params = json.loads(json.load(open(path + 'hparams.json')))

    gnn_model = GraphNet(params)
    gnn_model.load_state_dict(torch.load(path + 'trained_gnn.pms'))
    gnn_model.eval()

    model_name = '0063_1001'
    dataset, coefs_dict = pp.generate_dataset(model_name, dataset_params)

    test_train(gnn_model, model_name, dataset)
    test_rollout(gnn_model, model_name, dataset.lightgraphs[0], coefs_dict, do_plot = True, out_folder = '.')
