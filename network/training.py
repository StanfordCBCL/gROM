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
from datetime import datetime
import json

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def generate_gnn_model(params_dict):
    return GraphNet(params_dict)

def train_gnn_model(gnn_model, model_name, train_params):
    dataset, coefs_dict = pp.generate_dataset(model_name)
    num_examples = len(dataset)
    num_train = int(num_examples)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=10, drop_last=False)

    optimizer = torch.optim.Adam(gnn_model.parameters(),
                                 train_params['learning_rate'],
                                 weight_decay=train_params['weight_decay'])
    nepochs = train_params['nepochs']
    for epoch in range(nepochs):
        print('ep = ' + str(epoch))
        print('nepochs ' + str(nepochs))
        global_loss = 0
        count = 0
        for batched_graph in train_dataloader:
            pred = gnn_model(batched_graph,
                             batched_graph.ndata['n_features'].float()).squeeze()
            weight = torch.ones(pred.shape)
            # mask out values corresponding to boundary conditions
            weight[batched_graph.ndata['inlet_mask'],1] = 0
            weight[batched_graph.ndata['outlet_mask'],0] = 0
            loss = weighted_mse_loss(pred,
                                     torch.reshape(batched_graph.ndata['n_labels'].float(),
                                     pred.shape), weight)
            global_loss = global_loss + loss.detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = count + 1
        print('\tloss = ' + str(global_loss / count))

    return gnn_model, train_dataloader, global_loss / count

def plot_prediction(model, model_name, train_dataloader):
    it = iter(train_dataloader)
    batch = next(it)
    batched_graph = batch
    graph = dgl.unbatch(batched_graph)[0]

    true_graph = load_graphs('../dataset/data/' + model_name + '.grph')[0][0]
    times = pp.get_times(true_graph)
    times = times[50:]
    initial_pressure = pp.min_max(true_graph.ndata['pressure_' + str(times[0])],
                                  coefs_dict['pressure'])
    initial_flowrate = pp.min_max(true_graph.ndata['flowrate_' + str(times[0])],
                                  coefs_dict['flowrate'])
    graph = pp.set_state(graph, initial_pressure, initial_flowrate)

    for tind in range(len(times)-1):
        t = times[tind]
        tp1 = times[tind+1]

        next_pressure = true_graph.ndata['pressure_' + str(tp1)]
        next_flowrate = true_graph.ndata['flowrate_' + str(tp1)]
        np_normalized = pp.min_max(next_pressure, coefs_dict['pressure'])
        nf_normalized = pp.min_max(next_flowrate, coefs_dict['flowrate'])
        graph = pp.set_bcs(graph, np_normalized, nf_normalized)
        pred = model(graph, graph.ndata['n_features'].float()).squeeze()

        dp = pp.invert_min_max(pred[:,0].detach().numpy(), coefs_dict['dp'])
        prev_p = pp.invert_min_max(graph.ndata['pressure'].detach().numpy().squeeze(), coefs_dict['pressure'])

        p = prev_p + dp

        fig1 = plt.figure()
        ax1 = plt.axes()
        ax1.plot(p,'r')
        ax1.plot(next_pressure,'--b')

        dq = pp.invert_min_max(pred[:,1].detach().numpy(), coefs_dict['dq'])
        prev_q = pp.invert_min_max(graph.ndata['flowrate'].detach().numpy().squeeze(),
                                   coefs_dict['flowrate'])

        q = prev_q + dq

        fig2 = plt.figure()
        ax2 = plt.axes()
        ax2.plot(q,'r')
        ax2.plot(next_flowrate,'--b')

        new_pressure = pp.min_max(p, coefs_dict['pressure'])
        new_flowrate = pp.min_max(q, coefs_dict['flowrate'])
        graph = pp.set_state(graph, torch.unsqueeze(torch.from_numpy(new_pressure),1),
                                    torch.unsqueeze(torch.from_numpy(new_flowrate),1))
        # graph = pp.set_state(graph, np_normalized, nf_normalized)

        plt.show()

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        pass

def launch_training(model_name, params_dict,
                    train_params, plot_validation = True):
    create_directory('models')
    gnn_model = generate_gnn_model(params_dict)
    gnn_model, train_loader, loss = train_gnn_model(gnn_model,
                                                    model_name,
                                                    train_params)
    if (plot_validation):
        plot_prediction(gnn_model, model_name, train_loader)

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
    return gnn_model, loss

if __name__ == "__main__":
    params_dict = {'infeat_nodes': 8,
                   'infeat_edges': 5,
                   'latent_size': 16,
                   'out_size': 2,
                   'process_iterations': 1,
                   'hl_mlp': 2,
                   'normalize': True}
    train_params = {'learning_rate': 0.001,
                    'weight_decay': 0.0,
                    'nepochs': 30}
    main(sys.argv[1], params_dict, train_params)
