import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import dgl
import torch
import numpy as np
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset

def set_state(graph, pressure, flowrate):
    edges0 = graph.edges()[0]
    edges1 = graph.edges()[1]
    flowrate_edge = (flowrate[edges0] + flowrate[edges1]) / 2
    graph.ndata['pressure'] = pressure
    graph.ndata['flowrate'] = flowrate
    graph.edata['flowrate_edge'] = flowrate_edge
    graph.ndata['n_features'] = torch.cat((pressure, \
                                           flowrate, \
                                           graph.ndata['area'], \
                                           graph.ndata['node_type']), 1)
    graph.edata['e_features'] = torch.cat((graph.edata['position'], \
                                           flowrate_edge), 1)
    graph.ndata['n_labels'] = torch.cat((graph.ndata['dp'], \
                                         graph.ndata['dq']), 1)
    graph.edata['e_labels'] = graph.edata['dq_edge']
    return graph

def set_bcs(graph, next_pressure, next_flowrate):
    im = graph.ndata['inlet_mask']
    om = graph.ndata['outlet_mask']
    graph.ndata['pressure'][om] = next_pressure[om]
    graph.ndata['flowrate'][im] = next_flowrate[im]
    return graph

class DGL_Dataset(DGLDataset):
    def __init__(self, graphs):
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

            inlet_mask = new_graph.ndata['inlet_mask']
            outlet_mask = new_graph.ndata['outlet_mask']

            new_graph.ndata['pressure'] = graph.ndata['pressure_' + str(t)]
            new_graph.ndata['dp'] = graph.ndata['pressure_' + str(tp1)] - \
                                    graph.ndata['pressure_' + str(t)]
            new_graph.ndata['flowrate'] = graph.ndata['flowrate_' + str(t)]
            new_graph.ndata['dq'] = graph.ndata['flowrate_' + str(tp1)] - \
                                    graph.ndata['flowrate_' + str(t)]
            new_graph.edata['flowrate_edge_'] = graph.edata['flowrate_edge_' + str(t)]
            new_graph.edata['dq_edge'] = graph.edata['flowrate_edge_' + str(tp1)] - \
                                         graph.edata['flowrate_edge_' + str(t)]

            # overwrite boundary conditions
            new_graph.ndata['pressure'][outlet_mask] = graph.ndata['pressure_' + str(tp1)]
            new_graph.ndata['flowrate'][inlet_mask] = graph.ndata['flowrate_' + str(tp1)]

            out_graphs.append(new_graph)

    return out_graphs

def min_max(field, bounds):
    ncomponents = bounds[0].size
    if ncomponents == 1:
        return (field - bounds[0]) / (bounds[1] - bounds[0])
    for i in range(ncomponents):
        field[:,i] = (field[:,i] - bounds[0][i]) / (bounds[1][i] - bounds[0][i])
    return field

def invert_min_max(field, bounds):
    return bounds[0] + field * (bounds[1] - bounds[0])

def compute_min_max(graph, field):
    m = []
    M = []
    node_features = graph.ndata
    for feat in node_features:
        if field in feat:
            if len(m) == 0:
                m = np.min(graph.ndata[feat].numpy(), axis=0)
                M = np.max(graph.ndata[feat].numpy(), axis=0)
            else:
                m = np.minimum(np.min(graph.ndata[feat].numpy(), axis=0), m)
                M = np.maximum(np.max(graph.ndata[feat].numpy(), axis=0), M)

    edge_features = graph.edata
    for feat in edge_features:
        if field in feat:
            if len(m) == 0:
                m = np.min(graph.edata[feat].numpy(), axis=0)
                M = np.max(graph.edata[feat].numpy(), axis=0)
            else:
                m = np.minimum(np.min(graph.edata[feat].numpy(), axis=0), m)
                M = np.maximum(np.max(graph.edata[feat].numpy(), axis=0), M)

    if m.size == 1:
        m = float(m)
    if M.size == 1:
        M = float(M)

    return m, M

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

def normalize(graphs):
    norm_graphs = []
    fields = {'pressure', 'flowrate', 'area', 'position', 'dp', 'dq'}
    coefs_dict = {}
    for graph in graphs:
        cgraph = graph
        for field in fields:
            coefs = compute_min_max(cgraph, field)
            if field in coefs_dict:
                coefs_dict[field] = (np.minimum(coefs[0], coefs_dict[field][0]),
                                     np.maximum(coefs[1], coefs_dict[field][1]))
            else:
                coefs_dict[field] = coefs

    for graph in graphs:
        cgraph = graph
        cgraph = min_max_normalization(cgraph, fields, coefs_dict)

        norm_graphs.append(cgraph)

    return norm_graphs, coefs_dict

def generate_dataset(model_name):
    graphs = load_graphs('../dataset/data/' + model_name + '.grph')[0]
    graphs = create_single_timestep_graphs(graphs)
    graphs, coefs_dict = normalize(graphs)
    return DGL_Dataset(graphs), coefs_dict

if __name__ == "__main__":
    generate_dataset(sys.argv[1])
