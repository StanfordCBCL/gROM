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

def create_geometry(model_name, input_dir, sampling, remove_caps, points_to_keep = None, doresample = True):
    print('Create geometry: ' + model_name)
    soln = io.read_geo(input_dir + '/' + model_name + '.vtp').GetOutput()
    fields, _, p_array = io.get_all_arrays(soln, points_to_keep)
    
    
    p_array = np.zeros((20,3))
    p_array[:,0] = np.linspace(0,1,20)
    
    for field in fields:
        fields[field] = np.ones((20))
    
    return ResampledGeometry(Geometry(p_array), sampling, remove_caps, doresample), fields

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
            new_graph.ndata['pressure_' + str(t)] = torch.from_numpy((new_p).astype(DTYPE))
            new_graph.ndata['noise_p_' + str(t)] = torch.from_numpy((noise_p).astype(DTYPE))
            new_q = velocity[t]
            if len(graphs) != 0:
                noise_q = noise_q + np.random.normal(0, rate_noise, (nQ, 1)) * new_q
            new_graph.ndata['flowrate_' + str(t)] = torch.from_numpy((new_q).astype(DTYPE))
            new_graph.ndata['noise_q_' + str(t)] = torch.from_numpy((noise_q).astype(DTYPE))
        graphs.append(new_graph)

    return graphs

def generate_analytic(pressure, velocity, area):
    times = [t for t in pressure]
    times.sort()
    
    N = np.size(pressure[times[0]])

    xs = np.linspace(0, 2 * np.pi, N)

    # for i in range(0, N):
    #     pressure[times[0]][i] = np.sin(xs[i])
    #     velocity[times[0]][i] = xs[i] * xs[i]

    # for tin in range(1,len(times)):
    #     for n in range(1, N-1):
    #         pressure[times[tin]][n] = pressure[times[tin-1]][n] + 0.001 * area[n] * np.sin(velocity[times[tin-1]][n-1]) * np.cos(velocity[times[tin-1]][n+1])
    #         velocity[times[tin]][n] = velocity[times[tin-1]][n] + 0.001 * np.sqrt(area[n]) * np.cos((pressure[times[tin-1]][n-1] + pressure[times[tin-1]][n+1])/2)
    #         # pressure[times[tin]][n] = pressure[times[tin-1]][n] + 0.001 * area[n] * np.sin(velocity[times[tin-1]][n])
    #         # velocity[times[tin]][n] = velocity[times[tin-1]][n] + 0.001 * np.sqrt(area[n]) * np.cos(pressure[times[tin-1]][n])
    #     # pressure[times[tin]] = pressure[times[tin-1]] + 0.001 * area * np.sin(velocity[times[tin-1]])
    #     # velocity[times[tin]] = velocity[times[tin-1]] + 0.001 * np.sqrt(area) * np.cos(pressure[times[tin-1]])
    
    for tin in range(0, len(times)):
        for i in range(0, N):
            pressure[times[tin]][i] = np.sin(0.01 * tin) * np.cos(0.01 * tin)
            velocity[times[tin]][i] = np.cos(0.01 * tin) * 0.01 * tin
        
    return pressure, velocity


def generate_graphs(argv, dataset_params, input_dir, save = True):
    print('Generating_graphs with params ' + str(dataset_params))
    model_name = sys.argv[1]
    geo, fields = create_geometry(model_name, input_dir, 5, remove_caps = True,
                                  points_to_keep = 100, doresample = None)
    pressure, velocity = io.gather_pressures_velocities(fields)
    pressure, velocity, area = geo.generate_fields(pressure,
                                                   velocity,
                                                   fields['area'])

    pressure, velocity = generate_analytic(pressure, velocity, area)

    fixed_graph = create_fixed_graph(geo, area)
    graphs = add_fields(fixed_graph, pressure, velocity,
                        random_walks=dataset_params['random_walks'],
                        rate_noise=dataset_params['rate_noise'])
    if save:
        dgl.save_graphs('data/' + sys.argv[2], graphs)
    return graphs

if __name__ == "__main__":
    input_dir = 'vtps'
    dataset_params = {'random_walks': 0,
                      'rate_noise': 1e-3}
    generate_graphs(sys.argv, dataset_params, input_dir, True)
