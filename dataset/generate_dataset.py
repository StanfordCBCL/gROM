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

DTYPE = np.float32

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

    # find node type
    nnodes = nodes.shape[0]
    node_degree = np.zeros((nnodes))
    for j in range(0, nnodes):
        node_degree[j] = (np.count_nonzero(edg0 == j) + \
                          np.count_nonzero(edg1 == j))

    node_degree = np.array(node_degree)
    degrees = set()
    for j in range(0, nnodes):
        degrees.add(node_degree[j])

    # + 1 for boundary conditions (2 degrees nodes must be differentiated)
    node_type = np.zeros((nnodes,len(degrees) + 1))
    for j in range(0, nnodes):
        if node_degree[j] != 2:
            node_type[j,int(node_degree[j] / 2) - 2] = 1

    node_type[inlet_node, -2] = 1
    node_type[outlet_nodes, -1] = 1

    graph.ndata['area'] = torch.from_numpy(area.astype(DTYPE))
    graph.edata['position'] = torch.from_numpy(np.array(pos_feat).astype(DTYPE))
    graph.ndata['node_type'] = torch.from_numpy(node_type.astype(np.int))

    inlet_mask = np.zeros(nnodes)
    inlet_mask[inlet_node] = 1
    graph.ndata['inlet_mask'] = torch.from_numpy(inlet_mask.astype(np.int))

    outlet_mask = np.zeros(nnodes)
    outlet_mask[outlet_nodes] = 1
    graph.ndata['outlet_mask'] = torch.from_numpy(outlet_mask.astype(np.int))

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
        new_graph.ndata['area'] = torch.clone(graph.ndata['area'])
        new_graph.ndata['node_type'] = torch.clone(graph.ndata['node_type'])
        new_graph.edata['position'] = torch.clone(graph.edata['position'])
        new_graph.ndata['inlet_mask'] = torch.clone(graph.ndata['inlet_mask'])
        new_graph.ndata['outlet_mask'] = torch.clone(graph.ndata['outlet_mask'])
        noise_p = np.zeros((nP, 1))
        noise_q = np.zeros((nQ, 1))
        for t in times:
            new_p = pressure[t]
            if len(graphs) != 0:
                noise_p = noise_p + np.random.normal(0, rate_noise, (nP, 1)) * new_p
            new_graph.ndata['pressure_' + str(t)] = torch.from_numpy((new_p + noise_p).astype(DTYPE))

            new_q = velocity[t]
            if len(graphs) != 0:
                noise_q = noise_q + np.random.normal(0, rate_noise, (nQ, 1)) * new_q
            new_graph.ndata['flowrate_' + str(t)] = torch.from_numpy((new_q + noise_q).astype(DTYPE))
            new_graph.edata['flowrate_edge_' + str(t)] = torch.from_numpy((new_q[edges0] + new_q[edges1]).astype(DTYPE) / 2)
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
