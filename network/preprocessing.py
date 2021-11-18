import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("core/")

import dgl
import torch
import numpy as np
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset

class DGL_Dataset(DGLDataset):
    def __init__(self, graphs):
        self.graphs = graphs
        super().__init__(name='aorta')

    def process(self):
        for graph in self.graphs:
            graph.ndata['n_features'] = torch.cat((graph.ndata['pressure'], \
                                                   graph.ndata['flowrate'], \
                                                   graph.ndata['area'], \
                                                   graph.ndata['node_type']), 1)
            graph.edata['e_features'] = torch.cat((graph.edata['position'], \
                                                   graph.edata['flowrate_edge_']), 1)
            graph.ndata['n_labels'] = torch.cat((graph.ndata['dp'], \
                                                 graph.ndata['dq']), 1)
            graph.edata['e_labels'] = graph.edata['dq_edge']


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

        new_graph = dgl.graph((graph.edges()[0], graph.edges()[1]))
        new_graph.ndata['area'] = graph.ndata['area']
        new_graph.ndata['node_type'] = graph.ndata['node_type']
        new_graph.edata['position'] = graph.edata['position']

        ntimes = len(times)
        for tind in range(0, ntimes-1):
            t = times[tind]
            tp1 = times[tind+1]
            new_graph.ndata['pressure'] = graph.ndata['pressure_' + str(t)]
            new_graph.ndata['dp'] = graph.ndata['pressure_' + str(tp1)] - \
                                    graph.ndata['pressure_' + str(t)]
            new_graph.ndata['flowrate'] = graph.ndata['flowrate_' + str(t)]
            new_graph.ndata['dq'] = graph.ndata['flowrate_' + str(tp1)] - \
                                    graph.ndata['flowrate_' + str(t)]
            new_graph.edata['flowrate_edge_'] = graph.edata['flowrate_edge_' + str(t)]
            new_graph.edata['dq_edge'] = graph.edata['flowrate_edge_' + str(tp1)] - \
                                         graph.edata['flowrate_edge_' + str(t)]
            out_graphs.append(new_graph)

    return out_graphs

def min_max_normalization(graph, field):
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

    for feat in node_features:
        if field in feat:
            graph.ndata[feat] = (graph.ndata[feat] - m) / (M - m)

    for feat in edge_features:
        if field in feat:
            graph.edata[feat] = (graph.edata[feat] - m) / (M - m)

    if m.size == 1:
        m = float(m)
    if M.size == 1:
        M = float(M)

    return graph, m, M

def normalize(graphs):
    norm_graphs = []
    fields = {'pressure', 'flowrate', 'area', 'position', 'dp', 'dq'}
    coefs_dict = {}
    for graph in graphs:
        cgraph = graph
        for field in fields:
            out = min_max_normalization(cgraph, field)
            cgraph = out[0]
            coefs = out[1:]
            if field in coefs_dict:
                coefs_dict[field] = (np.minimum(coefs[0], coefs_dict[field][0]),
                                     np.maximum(coefs[1], coefs_dict[field][1]))
            else:
                coefs_dict[field] = coefs
        norm_graphs.append(cgraph)

    return norm_graphs, coefs_dict

def generate_dataset(model_name):
    graphs = load_graphs('../dataset/data/' + model_name + '.grph')[0]
    graphs = create_single_timestep_graphs(graphs)
    graphs, coefs_dict = normalize(graphs)
    return DGL_Dataset(graphs)

if __name__ == "__main__":
    generate_dataset(sys.argv[1])
