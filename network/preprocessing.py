import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("core/")

import dgl
import torch
from dgl.data.utils import load_graphs

# def min_max_normalization(graph, field):
#     m = torch.min(field)
#     M = torch.max(field)
#
#     for g in graphs:
#
#
#
#
# def normalize_data(graph, normalize_function):

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
            new_graph.ndata['d_pressure'] = graph.ndata['pressure_' + str(tp1)] - \
                                            graph.ndata['pressure_' + str(t)]
            new_graph.ndata['flowrate'] = graph.ndata['flowrate_' + str(t)]
            new_graph.ndata['d_flowrate'] = graph.ndata['flowrate_' + str(tp1)] - \
                                            graph.ndata['flowrate_' + str(t)]
            new_graph.edata['flowrate_edge_'] = graph.edata['flowrate_edge_' + str(t)]
            new_graph.edata['flowrate_edge_'] = graph.edata['flowrate_edge_' + str(tp1)] - \
                                                graph.edata['flowrate_edge_' + str(t)]
        out_graphs.append(new_graph)

def main(argv):
    model_name = sys.argv[1]
    graphs = load_graphs('../dataset/data/' + model_name + '.grph')[0]
    graphs = create_single_timestep_graphs(graphs)

    # fields = {'pressure', 'flowrate', 'area', 'position'}
    #
    # for graph in graphs:
    #     for field in fields:
    #         normalize_data(graph, field, min_max_normalization)

if __name__ == "__main__":
   main(sys.argv)
