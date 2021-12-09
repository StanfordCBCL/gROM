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

def set_state(graph, pressure, flowrate):
    # edges0 = graph.edges()[0]
    # edges1 = graph.edges()[1]
    # flowrate_edge = (flowrate[edges0] + flowrate[edges1]) / 2
    graph.ndata['pressure'] = pressure
    graph.ndata['flowrate'] = flowrate
    # graph.edata['flowrate_edge'] = flowrate_edge
    graph.ndata['n_features'] = torch.cat((pressure, \
                                           flowrate, \
                                           graph.ndata['area'], \
                                           graph.ndata['node_type']), 1)
    # graph.edata['e_features'] = torch.cat((graph.edata['position'], \
    #                                        flowrate_edge), 1)
    graph.edata['e_features'] = graph.edata['position']
    graph.ndata['n_labels'] = torch.cat((graph.ndata['dp'], \
                                         graph.ndata['dq']), 1)
    # graph.edata['e_labels'] = graph.edata['dq_edge']
    return graph

def set_bcs(graph, next_pressure, next_flowrate):
    im = np.where(graph.ndata['inlet_mask'].detach().numpy() == 1)[0]
    om = np.where(graph.ndata['outlet_mask'].detach().numpy() == 1)[0]
    graph.ndata['pressure'][om] = next_pressure[om]
    graph.ndata['flowrate'][im] = next_flowrate[im]
    return graph

class DGL_Dataset(DGLDataset):
    def __init__(self, graphs, resample_freq_timesteps = -1):
        if resample_freq_timesteps != -1:
            self.graphs = graphs[::resample_freq_timesteps]
        else:
            self.graphs = graphs
        super().__init__(name='dgl_dataset')

    def process(self):
        for graph in self.graphs:
            graph = set_state(graph, graph.ndata['pressure'],
                                     graph.ndata['flowrate'])

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

def get_times(graph):
    times = []

    features = graph.ndata
    for feature in features:
        if 'pressure' in feature:
            ind  = feature.find('_')
            times.append(float(feature[ind+1:]))
    times.sort()

    return times

def create_single_timestep_graphs(graphs):
    out_graphs = []
    for graph in graphs:
        times = get_times(graph)
        ntimes = len(times)
        for tind in range(0, ntimes-1):
            t = times[tind]
            tp1 = times[tind+1]

            new_graph = dgl.graph((graph.edges()[0], graph.edges()[1]))
            new_graph.ndata['area'] = torch.clone(graph.ndata['area'])
            new_graph.ndata['node_type'] = torch.clone(graph.ndata['node_type'])
            new_graph.edata['position'] = torch.clone(graph.edata['position'])
            new_graph.ndata['inlet_mask'] = torch.clone(graph.ndata['inlet_mask'])
            new_graph.ndata['outlet_mask'] = torch.clone(graph.ndata['outlet_mask'])


            new_graph.ndata['pressure'] = graph.ndata['pressure_' + str(t)] + \
                                          graph.ndata['noise_p_' + str(t)]
            new_graph.ndata['dp'] = graph.ndata['pressure_' + str(tp1)] - \
                                    graph.ndata['pressure_' + str(t)] - \
                                    graph.ndata['noise_p_' + str(t)]

            new_graph.ndata['flowrate'] = graph.ndata['flowrate_' + str(t)] + \
                                          graph.ndata['noise_q_' + str(t)]
            new_graph.ndata['dq'] = graph.ndata['flowrate_' + str(tp1)] - \
                                    graph.ndata['flowrate_' + str(t)] - \
                                    graph.ndata['noise_q_' + str(t)]

            # overwrite boundary conditions. This needs to be changed if bcs are
            # not perfect (e.g., resistance) to account for noise
            new_graph = set_bcs(new_graph, \
                                graph.ndata['pressure_' + str(tp1)], \
                                graph.ndata['flowrate_' + str(tp1)])

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
    node_features = graph.ndata
    for feat in node_features:
        for field in fields:
            if field in feat:
                if np.linalg.norm(np.min(graph.ndata[feat].detach().numpy()) - 0) > 1e-5 and \
                   np.linalg.norm(np.max(graph.ndata[feat].detach().numpy()) - 1) > 1e-5:
                       graph.ndata[feat] = min_max(graph.ndata[feat], bounds_dict[field])

    edge_features = graph.edata
    for feat in edge_features:
        for field in fields:
            if field in feat:
                if np.linalg.norm(np.min(graph.edata[feat].detach().numpy()) - 0) > 1e-5 and \
                   np.linalg.norm(np.max(graph.edata[feat].detach().numpy()) - 1) > 1e-5:
                       graph.edata[feat] = min_max(graph.edata[feat], bounds_dict[field])

    return graph

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
    node_features = graph.ndata
    for feat in node_features:
        for field in fields:
            if field in feat:
                graph.ndata[feat] = standardize(graph.ndata[feat], coeffs_dict[field])

    edge_features = graph.edata
    for feat in edge_features:
        for field in fields:
            if field in feat:
                graph.edata[feat] = standardize(graph.edata[feat], coeffs_dict[field])

    return graph

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
    node_features = graph.ndata
    for feat in node_features:
        if field in feat:
            if partial_list.size == 0:
                partial_list = graph.ndata[feat].detach().numpy()
            else:
                partial_list = np.concatenate((partial_list, graph.ndata[feat].detach().numpy()), axis = 0)

    edge_features = graph.edata
    for feat in edge_features:
        if field in feat:
            if partial_list.size == 0:
                partial_list = graph.edata[feat].detach().numpy()
            else:
                partial_list = np.concatenate((partial_list, graph.edata[feat].detach().numpy()), axis = 0)

    return partial_list

def normalize(graphs, type):
    norm_graphs = []
    coefs_dict = {}
    list_fields = {}
    coefs_dict['type'] = type
    fields = {'pressure', 'flowrate', 'area', 'position', 'dp', 'dq'}
    for field in fields:
        cur_list = np.zeros((0,0))
        for graph in graphs:
            cur_list = add_to_list(graph, field, cur_list)

        coefs_dict[field] = {'min': np.min(cur_list, axis=0),
                             'max': np.max(cur_list, axis=0),
                             'mean': np.mean(cur_list, axis=0),
                             'std': np.std(cur_list, axis=0)}

    for graph in graphs:
        cgraph = graph
        if type == 'min_max':
            cgraph = min_max_normalization(cgraph, fields, coefs_dict)
        if type == 'standard':
            cgraph = standard_normalization(cgraph, fields, coefs_dict)

        norm_graphs.append(cgraph)

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

    if dataset_params != None:
        return DGL_Dataset(graphs, dataset_params['resample_freq_timesteps']), \
                           coefs_dict
    else:
        return DGL_Dataset(graphs), coefs_dict
