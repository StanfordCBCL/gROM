import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append("../graphs")
sys.path.append("../graphs/core")

import dgl
import torch
import numpy as np
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset
import generate_graphs as gg
import copy
import random

def set_state(graph, state_dict):
    def per_node_type(node_type):
        graph.nodes[node_type].data['pressure'] = state_dict['pressure'][node_type]
        graph.nodes[node_type].data['flowrate'] = state_dict['flowrate'][node_type]
        if node_type == 'inner':
            graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure'], \
                                                                   graph.nodes[node_type].data['flowrate'], \
                                                                   graph.nodes[node_type].data['area'], \
                                                                   graph.nodes[node_type].data['node_type']), 1)
        else:
            graph.nodes[node_type].data['n_features'] = torch.cat((graph.nodes[node_type].data['pressure_next'], \
                                                                   graph.nodes[node_type].data['flowrate_next'], \
                                                                   graph.nodes[node_type].data['area']), 1)
        graph.nodes[node_type].data['n_labels'] = torch.cat((graph.nodes[node_type].data['dp'], \
                                                             graph.nodes[node_type].data['dq']), 1)

    def per_edge_type(edge_type):
        if edge_type == 'inner_to_inner':
            graph.edges[edge_type].data['e_features'] = graph.edges[edge_type].data['position']
        else:
            graph.edges[edge_type].data['e_features'] = torch.cat((graph.edges[edge_type].data['distance'][:,None],
                                                                   graph.edges[edge_type].data['physical_contiguous'][:,None]), 1)
    per_node_type('inner')
    per_node_type('inlet')
    per_node_type('outlet')
    per_edge_type('inner_to_inner')
    per_edge_type('in_to_inner')
    per_edge_type('out_to_inner')


def set_bcs(graph, state_dict):
    def per_node_type(node_type):
        graph.nodes[node_type].data['pressure_next'] = state_dict['pressure'][node_type]
        graph.nodes[node_type].data['flowrate_next'] = state_dict['flowrate'][node_type]
    per_node_type('inlet')
    per_node_type('outlet')

class DGL_Dataset(DGLDataset):
    def __init__(self, graphs = None):
        self.graphs = graphs
        super().__init__(name='dgl_dataset')
        # random.shuffle(self.graphs)

    def save_graphs(self, folder):
        dgl.save_graphs(folder + '/dataset.grph', self.graphs)

    def process(self):
        for graph in self.graphs:
            pressure_dict = {'inner': graph.nodes['inner'].data['pressure'],
                             'inlet': graph.nodes['inlet'].data['pressure'],
                             'outlet': graph.nodes['outlet'].data['pressure']}
            flowrate_dict = {'inner': graph.nodes['inner'].data['flowrate'],
                             'inlet': graph.nodes['inlet'].data['flowrate'],
                             'outlet': graph.nodes['outlet'].data['flowrate']}
            state_dict = {'pressure': pressure_dict, 'flowrate': flowrate_dict}
            set_state(graph, state_dict)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

def get_times(graph):
    times = []

    features = graph.nodes['inner'].data
    for feature in features:
        if 'pressure' in feature:
            ind  = feature.find('_')
            times.append(float(feature[ind+1:]))
    times.sort()

    return times

def free_fields(graph, times):
    def per_node_type(node_type):
        for t in times:
            del(graph.nodes[node_type].data['pressure_' + str(t)])
            del(graph.nodes[node_type].data['noise_p_' + str(t)])
            del(graph.nodes[node_type].data['flowrate_' + str(t)])
            del(graph.nodes[node_type].data['noise_q_' + str(t)])
    per_node_type('inner')
    per_node_type('inlet')
    per_node_type('outlet')

def set_timestep(targetgraph, allgraph, t, tp1):
    def per_node_type(node_type):
        targetgraph.nodes[node_type].data['pressure'] = allgraph.nodes[node_type].data['pressure_' + str(t)] + \
                                                        allgraph.nodes[node_type].data['noise_p_' + str(t)]
        targetgraph.nodes[node_type].data['dp'] = allgraph.nodes[node_type].data['pressure_' + str(tp1)] - \
                                allgraph.nodes[node_type].data['pressure_' + str(t)] - \
                                allgraph.nodes[node_type].data['noise_p_' + str(t)]

        targetgraph.nodes[node_type].data['flowrate'] = allgraph.nodes[node_type].data['flowrate_' + str(t)] + \
                                      allgraph.nodes[node_type].data['noise_q_' + str(t)]
        targetgraph.nodes[node_type].data['dq'] = allgraph.nodes[node_type].data['flowrate_' + str(tp1)] - \
                                allgraph.nodes[node_type].data['flowrate_' + str(t)] - \
                                allgraph.nodes[node_type].data['noise_q_' + str(t)]

        targetgraph.nodes[node_type].data['pressure_next'] = allgraph.nodes[node_type].data['pressure_' + str(tp1)]
        targetgraph.nodes[node_type].data['flowrate_next'] = allgraph.nodes[node_type].data['flowrate_' + str(tp1)]

    per_node_type('inner')
    per_node_type('inlet')
    per_node_type('outlet')

    # we add also current time to graph(for debugging()
    targetgraph.nodes['inlet'].data['time'] = torch.from_numpy(np.array([t]))

def create_single_timestep_graphs(graphs):
    out_graphs = []
    for graph in graphs:
        times = get_times(graph)
        graph_no_fields = copy.deepcopy(graph)
        free_fields(graph_no_fields, times)
        ntimes = len(times)
        for tind in range(0, ntimes-1):
            t = times[tind]
            tp1 = times[tind+1]

            new_graph = copy.deepcopy(graph_no_fields)
            set_timestep(new_graph, graph, t, tp1)


            out_graphs.append(new_graph)

    return out_graphs

def min_max(field, bounds):
    ncomponents = bounds['min'].size
    if ncomponents == 1:
        return (field - bounds['min']) / (bounds['max'] - bounds['min'])
    for i in range(ncomponents):
        field[:,i] = (field[:,i] - bounds['min'][i]) / (bounds['max'][i] - bounds['min'][i])
    return field

def invert_min_max(field, bounds):
    return bounds['min'] + field * (bounds['max'] - bounds['min'])

def min_max_normalization(graph, fields, bounds_dict):
    def per_node_type(node_type):
        node_features = graph.nodes[node_type].data
        for feat in node_features:
            for field in fields:
                if field in feat:
                    if np.linalg.norm(np.min(graph.nodes[node_type].data[feat].detach().numpy()) - 0) > 1e-5 and \
                       np.linalg.norm(np.max(graph.nodes[node_type].data[feat].detach().numpy()) - 1) > 1e-5:
                           graph.nodes[node_type].data[feat] = min_max(graph.nodes[node_type].data[feat], bounds_dict[field])

    def per_edge_type(edge_type):
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            for field in fields:
                if field in feat:
                    if np.linalg.norm(np.min(graph.edges[edge_type].data[feat].detach().numpy()) - 0) > 1e-5 and \
                       np.linalg.norm(np.max(graph.edges[edge_type].data[feat].detach().numpy()) - 1) > 1e-5:
                           graph.edges[edge_type].data[feat] = min_max(graph.edges[edge_type].data[feat], bounds_dict[field])

    per_node_type('inner')
    per_node_type('inlet')
    per_node_type('outlet')
    per_edge_type('inner_to_inner')
    per_edge_type('in_to_inner')
    per_edge_type('out_to_inner')

def standardize(field, coeffs):
    ncomponents = coeffs['mean'].size
    if ncomponents == 1:
        return (field - coeffs['mean']) / coeffs['std']
    for i in range(ncomponents):
        field[:,i] = (field[:,i] - coeffs['mean'][i]) / coeffs['std'][i]
    return field

def invert_standardize(field, coeffs):
    return coeffs['mean'] + field * coeffs['std']

def standard_normalization(graph, fields, coeffs_dict):
    def per_node_type(node_type):
        node_features = graph.nodes[node_type].data
        for feat in node_features:
            for field in fields:
                if field in feat:
                    graph.nodes[node_type].data[feat] = standardize(graph.nodes[node_type].data[feat], coeffs_dict[field])

    def per_edge_type(edge_type):
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            for field in fields:
                if field in feat:
                    graph.edges[edge_type].data[feat] = standardize(graph.edges[edge_type].data[feat], coeffs_dict[field])

    per_node_type('inner')
    per_node_type('inlet')
    per_node_type('outlet')
    per_edge_type('inner_to_inner')
    per_edge_type('in_to_inner')
    per_edge_type('out_to_inner')

def normalize_function(field, field_name, coefs_dict):
    if coefs_dict['type'] == 'min_max':
        return min_max(field, coefs_dict[field_name])
    elif coefs_dict['type'] == 'standard':
        return standardize(field, coefs_dict[field_name])
    return []

def invert_normalize_function(field, field_name, coefs_dict):
    if coefs_dict['type'] == 'min_max':
        return invert_min_max(field, coefs_dict[field_name])
    if coefs_dict['type'] == 'standard':
        return invert_standardize(field, coefs_dict[field_name])
    return []

def add_to_list(graph, field, partial_list):
    def per_node_type(node_type, partial_list):
        node_features = graph.nodes[node_type].data
        for feat in node_features:
            if field in feat:
                value = graph.nodes[node_type].data[feat].detach().numpy()
                if (len(value.shape) == 1):
                    value = np.expand_dims(value, axis = 1)
                if partial_list.size == 0:
                    partial_list = value
                else:
                    partial_list = np.concatenate((partial_list,value), axis = 0)
        return partial_list

    def per_edge_type(edge_type, partial_list):
        edge_features = graph.edges[edge_type].data
        for feat in edge_features:
            if field in feat:
                value = graph.edges[edge_type].data[feat].detach().numpy()
                if (len(value.shape) == 1):
                    value = np.expand_dims(value, axis = 1)
                if partial_list.size == 0:
                    partial_list = value
                else:
                    partial_list = np.concatenate((partial_list, value), axis = 0)
        return partial_list

    partial_list = per_node_type('inner', partial_list)
    partial_list = per_node_type('inlet', partial_list)
    partial_list = per_node_type('outlet', partial_list)
    partial_list = per_edge_type('inner_to_inner', partial_list)
    partial_list = per_edge_type('in_to_inner', partial_list)
    partial_list = per_edge_type('out_to_inner', partial_list)

    return partial_list

def compute_statistics(graphs, fields, coefs_dict):
    for field in fields:
        cur_list = np.zeros((0,0))
        for graph in graphs:
            cur_list = add_to_list(graph, field, cur_list)

        coefs_dict[field] = {'min': np.min(cur_list, axis=0),
                             'max': np.max(cur_list, axis=0),
                             'mean': np.mean(cur_list, axis=0),
                             'std': np.std(cur_list, axis=0)}

        ncoefs = coefs_dict[field]['std'].shape[0]
        for i in range(ncoefs):
            if coefs_dict[field]['std'][i] < 1e-12:
                coefs_dict[field]['std'][i] = 1
    return coefs_dict

def normalize_graphs(graphs, fields, coefs_dict):
    norm_graphs = []

    for graph in graphs:
        if type == 'min_max':
            min_max_normalization(graph, fields, coefs_dict)
        if type == 'standard':
            standard_normalization(graph, fields, coefs_dict)

        norm_graphs.append(graph)

    return norm_graphs

def normalize(graphs, type):
    coefs_dict = {}
    coefs_dict['type'] = type
    fields = {'pressure', 'flowrate', 'area', 'position', 'dp', 'dq', 'distance'}

    coefs_dict = compute_statistics(graphs, fields, coefs_dict)
    norm_graphs = normalize_graphs(graphs, fields, coefs_dict)

    return norm_graphs, coefs_dict

def generate_dataset(model_name, dataset_params = None):
    if dataset_params == None:
        graphs = load_graphs('../graphs/data/' + model_name + '.grph')[0]
    else:
        graphs = gg.generate_graphs(model_name,
                                    dataset_params,
                                    '../graphs/vtps',
                                    False)

    normalization_type = 'standard'
    if dataset_params != None:
        normalization_type = dataset_params['normalization']

    graphs = create_single_timestep_graphs(graphs)
    graphs, coefs_dict = normalize(graphs, normalization_type)

    return DGL_Dataset(graphs), coefs_dict
