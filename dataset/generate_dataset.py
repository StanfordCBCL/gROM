import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("core/")

import matplotlib.pyplot as plt
import io_utils as io
from geometry import Geometry
from resampled_geometry import ResampledGeometry
import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def create_geometry(model_name, sampling):
    print('Create geometry: ' + model_name)
    soln = io.read_geo('vtps/' + model_name + '.vtp').GetOutput()
    fields, _, p_array = io.get_all_arrays(soln)
    return ResampledGeometry(Geometry(p_array), sampling), fields

def create_fixed_graph(geometry, area):
    nodes, edges, lengths, inlet_node, outlet_nodes = geometry.generate_nodes()

    graph = dgl.graph((edges[:,0], edges[:,1]))
    graph = dgl.to_bidirected(graph)

    pos_feat = []

    edg0 = graph.edges()[0]
    edg1 = graph.edges()[1]
    N = edg0.shape[0]
    for j in range(0, N):
        diff = nodes[edg1[j],:] - nodes[edg0[j],:]
        diff = np.hstack((diff, np.linalg.norm(diff)))
        pos_feat.append(diff)

    graph.ndata['area'] = torch.from_numpy(area)
    graph.edata['position'] = torch.from_numpy(np.array(pos_feat))

    print('Graph generated:')
    print(' n. nodes = ' + str(nodes.shape[0]))
    print(' n. edges = ' + str(N))

    return graph

def add_fields(graph, pressure, velocity, random_walks, rate_noise):
    print('Writing fields:')
    graphs = []
    times = [t for t in pressure]
    times.sort()
    nP = pressure[times[0]].shape[0]
    nQ = velocity[times[0]].shape[0]
    edges0 = graph.edges()[0]
    edges1 = graph.edges()[1]
    print('  n. times = ' + str(len(times)))
    while len(graphs) < random_walks + 1:
        print('  writing graph n. ' + str(len(graphs) + 1))
        new_graph = dgl.graph((graph.edges()[0], graph.edges()[1]))
        new_graph.ndata['area'] = graph.ndata['area']
        new_graph.edata['position'] = graph.edata['position']
        noise_p = np.zeros((nP, 1))
        noise_q = np.zeros((nQ, 1))
        for t in times:
            new_p = pressure[t]
            if len(graphs) != 0:
                noisep = noise_p + np.random.normal(0, rate_noise, (nP, 1)) * new_p
            new_graph.ndata['pressure_' + str(t)] = torch.from_numpy(new_p + noise_p)

            new_q = velocity[t]
            if len(graphs) != 0:
                noiseq = noise_q + np.random.normal(0, rate_noise, (nQ, 1)) * new_q
            new_graph.ndata['flowrate_' + str(t)] = torch.from_numpy(new_q + noise_q)
            new_graph.edata['flowrate_edge_' + str(t)] = torch.from_numpy((new_q[edges0] + new_q[edges1]) / 2)
        graphs.append(new_graph)

    return graphs

def main(argv):
    model_name = sys.argv[1]
    geo, fields = create_geometry(model_name, 10)
    pressure, velocity = io.gather_pressures_velocities(fields)
    pressure, velocity, area = geo.generate_fields(pressure,
                                                   velocity,
                                                   fields['area'])
    fixed_graph = create_fixed_graph(geo, area)
    graphs = add_fields(fixed_graph, pressure, velocity,
                        random_walks=9,
                        rate_noise=1e-4)
    dgl.save_graphs('data/' + sys.argv[2], graphs)

if __name__ == "__main__":
   main(sys.argv)
