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
import copy

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

def convert_nodes_to_heterogeneous(nodes, edges, inlet_node, outlet_nodes):
    inlet_edge = np.where(edges == inlet_node)
    row = inlet_edge[0]
    if inlet_edge[1] == 0:
        inlet_edge = np.array([edges[row,0], edges[row,1]]).transpose()
    else: 
        inlet_edge = np.array([edges[row,1], edges[row,0]]).transpose()
    edges = np.delete(edges,row, axis=0)
    
    outlet_edges = []
    for outlet_node in outlet_nodes:
        outlet_edge = np.where(edges == outlet_node)
        row = outlet_edge[0]
        if outlet_edge[1] == 0:
            outlet_edge = [edges[row,0], edges[row,1]]
        else:
            outlet_edge = [edges[row,1], edges[row,0]]
        outlet_edges.append(outlet_edge)
        edges = np.delete(edges,row, axis=0)

    outlet_edges = np.array(outlet_edges).squeeze(axis = 2)
    
    inlet_original = np.copy(inlet_edge)
    outlets_original = np.copy(outlet_edges)
    edges_original = np.copy(edges)
    
    # change numbering
    offset = np.min(edges)
    inlet_edge[:,1] = inlet_edge[:,1] - offset
    inlet_edge[0,0] = 0
    outlet_edges[:,1] = outlet_edges[:,1] - offset
    for j in range(outlet_edges.shape[0]):
        outlet_edges[0] = j
    edges = edges - offset
    
    # transform inner graph to bidirected
    edges = np.concatenate((edges,np.array([edges[:,1],edges[:,0]]).transpose()),axis = 0)
    edges_original = np.concatenate((edges_original,np.array([edges_original[:,1],edges_original[:,0]]).transpose()),axis = 0)

    return inlet_edge, outlet_edges, edges, inlet_original, outlets_original, edges_original
    

def create_fixed_graph(geometry, area):
    nodes, edges, lengths, inlet_node, outlet_nodes = geometry.generate_nodes()
    
    inlet_edge, outlet_edges, edges, \
    inlet_original, outlets_original, edges_original = convert_nodes_to_heterogeneous(nodes, edges, inlet_node, outlet_nodes)
    
    graph_data = {('inlet', 'in_to_inner', 'inner'): (inlet_edge[:,0], inlet_edge[:,1]),
                  ('inner', 'inner_to_inner', 'inner'): (edges[:,0],edges[:,1]),
                  ('outlet', 'out_to_inner', 'inner'): (outlet_edges[:,0],outlet_edges[:,1])}

    graph = dgl.heterograph(graph_data)
    # graph = dgl.to_bidirected(graph)
    
    
    # compute position for inner edges
    pos_feat = []

    edg0 = edges_original[:,0]
    edg1 = edges_original[:,1]
    N = edg0.shape[0]
    for j in range(0, N):
        diff = nodes[edg1[j],:] - nodes[edg0[j],:]
        diff = np.hstack((diff, np.linalg.norm(diff)))
        pos_feat.append(diff)
        
    graph.edges['inner_to_inner'].data['position'] = torch.from_numpy(np.array(pos_feat).astype(DTYPE))
    
    # compute position for inlet edge
    pos_feat = []
    edg0 = inlet_original[:,0]
    edg1 = inlet_original[:,1]
    diff = nodes[edg1[0],:] - nodes[edg0[0],:]
    diff = np.hstack((diff, np.linalg.norm(diff)))
    pos_feat.append(diff)
    
    graph.edges['in_to_inner'].data['position'] = torch.from_numpy(np.array(pos_feat).astype(DTYPE))
    
    # compute position for outer edges
    pos_feat = []
    
    edg0 = outlets_original[:,0]
    edg1 = outlets_original[:,1]
    N = edg0.shape[0]
    for j in range(0, N):
        diff = nodes[edg1[j],:] - nodes[edg0[j],:]
        diff = np.hstack((diff, np.linalg.norm(diff)))
        pos_feat.append(diff)
       
    graph.edges['out_to_inner'].data['position'] = torch.from_numpy(np.array(pos_feat).astype(DTYPE))

    # find inner node type
    edg0 = edges[:,0]
    edg1 = edges[:,1]
    # inner edges are bidirectional => /2
    nnodes = int(edges.shape[0] / 2) + 1
    node_degree = np.zeros((nnodes + 1))
    for j in range(0, nnodes + 1):
        node_degree[j] = (np.count_nonzero(edg0 == j) + \
                          np.count_nonzero(edg1 == j))

    node_degree = np.array(node_degree)
    degrees = set()
    for j in range(0, nnodes):
        degrees.add(node_degree[j])

    node_type = np.zeros((nnodes,len(degrees)))
    for j in range(0, nnodes):
        node_type[j,int(node_degree[j] / 2) - 1] = 1

    graph.nodes['inner'].data['node_type'] = torch.from_numpy(node_type.astype(int))
    
    # graph.ndata['area'] = torch.from_numpy(area.astype(DTYPE))\
        
    # set global mask and area
    nnodes = nodes.shape[0]
    indices = np.arange(nnodes)
    indices = np.delete(indices, [inlet_node] + outlet_nodes)
    graph.nodes['inner'].data['global_mask'] = torch.from_numpy(indices.astype(DTYPE))
    graph.nodes['inner'].data['area'] = torch.from_numpy(area[indices].astype(DTYPE))
    
    # set area inner
    graph.nodes['inlet'].data['global_mask'] = torch.from_numpy(np.array([inlet_node]).astype(DTYPE))
    areainlet = area[inlet_node]
    if (len(areainlet.shape) == 1):
        areainlet = np.expand_dims(areainlet, axis = 1)
    graph.nodes['inlet'].data['area'] = torch.from_numpy(areainlet.astype(DTYPE))
    
    graph.nodes['outlet'].data['global_mask'] = torch.from_numpy(np.array(outlet_nodes).astype(DTYPE))
    areaoutlet = area[outlet_nodes]
    if (len(areaoutlet.shape) == 1):
        areaoutlet = np.expand_dims(areaoutlet, axis = 1)
    graph.nodes['outlet'].data['area'] = torch.from_numpy(areaoutlet.astype(DTYPE))
    

    print('Graph generated:')
    print(' n. nodes = ' + str(nodes.shape[0]))
    print(' n. inner edges = ' + str(edges.shape[0]))
    print(' n. inlets = ' + str(inlet_edge.shape[0]))
    print(' n. outlets = ' + str(outlet_edges.shape[0]))

    return graph

def set_field(graph, name_field, field):
    def set_in_node(node_type):
        mask = graph.nodes[node_type].data['global_mask'].detach().numpy().astype(int)
        masked_field = torch.from_numpy(field[mask].astype(DTYPE))
        graph.nodes[node_type].data[name_field] = masked_field
    set_in_node('inner')
    set_in_node('inlet')
    set_in_node('outlet')

def add_fields(graph, pressure, velocity, random_walks, rate_noise):
    print('Writing fields:')
    graphs = []
    times = [t for t in pressure]
    times.sort()
    nP = pressure[times[0]].shape[0]
    nQ = velocity[times[0]].shape[0]
    print('  n. times = ' + str(len(times)))
    while len(graphs) < random_walks + 1:
        print('  writing graph n. ' + str(len(graphs) + 1))
        new_graph = copy.deepcopy(graph)
        noise_p = np.zeros((nP, 1))
        noise_q = np.zeros((nQ, 1))
        for t in times:
            new_p = pressure[t]
            if len(graphs) != 0:
                noise_p = noise_p + np.random.normal(0, rate_noise, (nP, 1)) * new_p
            set_field(new_graph, 'pressure_' + str(t), new_p)
            set_field(new_graph, 'noise_p_' + str(t), noise_p)
            new_q = velocity[t]
            if len(graphs) != 0:
                noise_q = noise_q + np.random.normal(0, rate_noise, (nQ, 1)) * new_q
            set_field(new_graph, 'flowrate_' + str(t), new_q)
            set_field(new_graph, 'noise_q_' + str(t), noise_q)
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
