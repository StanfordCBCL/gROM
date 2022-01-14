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
from scipy import interpolate
from matplotlib import animation

DTYPE = np.float32

def create_geometry(model_name, input_dir, sampling, remove_caps, points_to_keep = None, doresample = True):
    print('Create geometry: ' + model_name)
    soln = io.read_geo(input_dir + '/' + model_name + '.vtp').GetOutput()
    fields, _, p_array = io.get_all_arrays(soln, points_to_keep)

    return ResampledGeometry(Geometry(p_array), sampling, remove_caps, doresample), fields

def convert_nodes_to_heterogeneous(nodes, edges, inlet_index, outlet_indices):
    # Dijkstra's algorithm
    def dijkstra_algorithm(nodes, edges, index):
        # make edges bidirectional for simplicity
        nnodes = nodes.shape[0]
        tovisit = np.arange(0,nnodes)
        dists = np.ones((nnodes)) * np.infty
        prevs = np.ones((nnodes)) * (-1)
        b_edges = np.concatenate((edges, np.array([edges[:,1],edges[:,0]]).transpose()), axis = 0)

        dists[index] = 0
        while len(tovisit) != 0:

            minindex = -1
            minlen = np.infty
            for iinde in range(len(tovisit)):
                if dists[tovisit[iinde]] < minlen:
                    minindex = iinde
                    minlen = dists[tovisit[iinde]]

            curindex = tovisit[minindex]
            tovisit = np.delete(tovisit, minindex)

            # find neighbors of curindex
            inb = b_edges[np.where(b_edges[:,0] == curindex)[0],1]

            for neib in inb:
                if np.where(tovisit == neib)[0].size != 0:
                    alt = dists[curindex] + np.linalg.norm(nodes[curindex,:] - nodes[neib,:])
                    if alt < dists[neib]:
                        dists[neib] = alt
                        prevs[neib] = curindex
        return dists, prevs

    nnodes = nodes.shape[0]
    nninner_nodes = nnodes - len([inlet_index] + outlet_indices)

    # create inner mask from local to global
    indices = np.arange(nnodes)
    inner_mask = np.delete(indices, [inlet_index] + outlet_indices)

    # process inlet
    indices = np.arange(nnodes)
    inlet_mask = np.array([indices[inlet_index]])
    inlet_edges = np.zeros((nninner_nodes,2))
    inlet_edges[:,1] = np.arange(nninner_nodes)
    distances_inlet, _ = dijkstra_algorithm(nodes, edges, inlet_index)
    distances_inlet = np.delete(distances_inlet, [inlet_index] + outlet_indices)
    inlet_physical_contiguous = np.zeros(nninner_nodes)
    for iedg in range(edges.shape[0]):
        if edges[iedg,0] == inlet_index:
            inlet_physical_contiguous[np.where(inner_mask == edges[iedg,1])[0]] = 1
            break
        if edges[iedg,1] == inlet_index:
            inlet_physical_contiguous[np.where(inner_mask == edges[iedg,0])[0]] = 1
            break
    inlet_dict = {'edges': inlet_edges.astype(int), \
                  'distance': distances_inlet, \
                  'x': np.expand_dims(nodes[inlet_index,:],axis=0), \
                  'mask': inlet_mask, \
                  'physical_contiguous': inlet_physical_contiguous.astype(int)}

    # process outlets
    indices = np.arange(nnodes)
    outlet_mask = indices[outlet_indices]
    outlet_edges = np.zeros((0,2))
    distances_outlets = np.zeros((0))
    outlet_physical_contiguous = np.zeros((0))
    for out_index in range(len(outlet_indices)):
        curoutedge = np.copy(inlet_edges)
        curoutedge[:,0] = out_index
        outlet_edges = np.concatenate((outlet_edges, curoutedge), axis = 0)
        curdistances, _ = dijkstra_algorithm(nodes, edges, outlet_indices[out_index])
        curdistances = np.delete(curdistances, [inlet_index] + outlet_indices)
        distances_outlets = np.concatenate((distances_outlets, curdistances))
        cur_opc = np.zeros(nninner_nodes).astype(int)
        for iedg in range(edges.shape[0]):
            if edges[iedg,0] == outlet_indices[out_index]:
                cur_opc[np.where(inner_mask == edges[iedg,1])[0]] = 1
                break
            if edges[iedg,1] == outlet_indices[out_index]:
                cur_opc[np.where(inner_mask == edges[iedg,0])[0]] = 1
                break
        outlet_physical_contiguous = np.concatenate((outlet_physical_contiguous, cur_opc))

    # we select the edges and properties such that each inner node is only connected to one outlet
    # (based on smaller distance)
    single_connection_mask = []
    for inod in range(nninner_nodes):
        mindist = np.amin(distances_outlets[inod::nninner_nodes])
        indx = np.where(np.abs(distances_outlets - mindist) < 1e-14)[0]
        single_connection_mask.append(int(indx))


    outlet_dict = {'edges': outlet_edges[single_connection_mask,:].astype(int), \
                   'distance': distances_outlets[single_connection_mask], \
                   'x': nodes[outlet_indices,:], \
                   'mask': outlet_mask, \
                   'physical_contiguous': outlet_physical_contiguous[single_connection_mask].astype(int)}

    # renumber edges
    rowstodelete = []
    for irow in range(edges.shape[0]):
        for bcindex in [inlet_index] + outlet_indices:
            if bcindex in edges[irow,:]:
                rowstodelete.append(irow)

    edges = np.delete(edges, rowstodelete, axis = 0)
    edges_c = np.copy(edges)
    edges_c2 = np.copy(edges)
    for nodein in range(nninner_nodes):
        minind = np.amin(edges_c)
        indices = np.where(edges == minind)

        for idx in range(indices[0].shape[0]):
            edges[indices[0][idx],indices[1][idx]] = nodein
            edges_c[indices[0][idx],indices[1][idx]] = 1e9

    # make it bidirectional
    edges = np.concatenate((edges,np.array([edges[:,1],edges[:,0]]).transpose()),axis = 0)

    inner_nodes = np.delete(nodes, [inlet_index] + outlet_indices, axis = 0)

    nedges = edges.shape[0]
    inner_pos = np.zeros((nedges, 4))
    for iedg in range(nedges):
        inner_pos[iedg,0:3] = inner_nodes[edges[iedg,1],:] - inner_nodes[edges[iedg,0],:]
        inner_pos[iedg,3] = np.linalg.norm(inner_pos[iedg,0:2])

    inner_dict = {'edges': edges, 'position': inner_pos, 'x': inner_nodes, 'mask': inner_mask}

    return inner_dict, inlet_dict, outlet_dict

def create_fixed_graph(geometry, area):
    nodes, edges, _, inlet_index, outlet_indices = geometry.generate_nodes()

    inner_dict, inlet_dict, outlet_dict = \
        convert_nodes_to_heterogeneous(nodes, edges, inlet_index, outlet_indices)

    graph_data = {('inner', 'inner_to_inner', 'inner'): \
                  (inner_dict['edges'][:,0], inner_dict['edges'][:,1]),
                  ('inlet', 'in_to_inner', 'inner'): \
                  (inlet_dict['edges'][:,0], inlet_dict['edges'][:,1]), \
                  ('outlet', 'out_to_inner', 'inner'): \
                  (outlet_dict['edges'][:,0],outlet_dict['edges'][:,1]), \
                  ('params', 'dummy', 'params'): \
                  (np.array([0]), np.array([0]))}

    graph = dgl.heterograph(graph_data)

    graph.edges['inner_to_inner'].data['position'] = \
                        torch.from_numpy(inner_dict['position'].astype(DTYPE))
    graph.edges['inner_to_inner'].data['edges'] = \
                        torch.from_numpy(inner_dict['edges'])
    graph.edges['in_to_inner'].data['distance'] = \
                        torch.from_numpy(inlet_dict['distance'].astype(DTYPE))
    graph.edges['in_to_inner'].data['physical_contiguous'] = \
                        torch.from_numpy(inlet_dict['physical_contiguous'])
    graph.edges['in_to_inner'].data['edges'] = \
                        torch.from_numpy(inlet_dict['edges'])
    graph.edges['out_to_inner'].data['distance'] = \
                        torch.from_numpy(outlet_dict['distance'].astype(DTYPE))
    graph.edges['out_to_inner'].data['physical_contiguous'] = \
                        torch.from_numpy(outlet_dict['physical_contiguous'])
    graph.edges['out_to_inner'].data['edges'] = \
                        torch.from_numpy(outlet_dict['edges'])

    # find inner node type
    edg0 = inner_dict['edges'][:,0]
    edg1 = inner_dict['edges'][:,1]
    # inner edges are bidirectional => /2
    nnodes = np.max(inner_dict['edges']) + 1
    node_degree = np.zeros((nnodes))
    for j in range(0, nnodes):
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
    graph.nodes['inner'].data['x'] = torch.from_numpy(inner_dict['x'])
    graph.nodes['inner'].data['global_mask'] = torch.from_numpy(inner_dict['mask'])
    graph.nodes['inner'].data['area'] = torch.from_numpy(area[inner_dict['mask']].astype(DTYPE))

    graph.nodes['inlet'].data['global_mask'] = torch.from_numpy(inlet_dict['mask'])
    graph.nodes['inlet'].data['area'] = torch.from_numpy(area[inlet_dict['mask']].astype(DTYPE))
    graph.nodes['inlet'].data['x'] = torch.from_numpy(inlet_dict['x'])

    graph.nodes['outlet'].data['global_mask'] = torch.from_numpy(outlet_dict['mask'])
    graph.nodes['outlet'].data['area'] = torch.from_numpy(area[outlet_dict['mask']].astype(DTYPE))
    print(outlet_dict['x'].shape)
    graph.nodes['outlet'].data['x'] = torch.from_numpy(outlet_dict['x'])

    print('Graph generated:')
    print(' n. nodes = ' + str(nodes.shape[0]))
    print(' n. inner edges = ' + str(edges.shape[0]))
    print(' n. inlet edges = ' + str(inlet_dict['edges'].shape[0]))
    print(' n. outlet edges = ' + str(outlet_dict['edges'].shape[0]))

    return graph

def set_field(graph, name_field, field):
    def set_in_node(node_type):
        mask = graph.nodes[node_type].data['global_mask'].detach().numpy().astype(int)
        masked_field = torch.from_numpy(field[mask].astype(DTYPE))
        graph.nodes[node_type].data[name_field] = masked_field
    set_in_node('inner')
    set_in_node('inlet')
    set_in_node('outlet')

def add_fields(graph, pressure, velocity):
    print('Writing fields:')
    graphs = []
    times = [t for t in pressure]
    times.sort()
    nP = pressure[times[0]].shape[0]
    nQ = velocity[times[0]].shape[0]
    print('  n. times = ' + str(len(times)))

    newgraph = copy.deepcopy(graph)

    for t in range(len(times)):
        set_field(newgraph, 'pressure_' + str(t), pressure[times[t]])
        set_field(newgraph, 'flowrate_' + str(t), velocity[times[t]])

    newgraph.nodes['params'].data['times'] = \
                        torch.from_numpy(np.expand_dims(np.array(times),axis=0))

    return newgraph

def generate_analytic(pressure, velocity, area):
    times = [t for t in pressure]
    times.sort()

    N = np.size(pressure[times[0]])

    xs = np.linspace(0, 2 * np.pi, N)

    T = len(times)
    for tin in range(0, T):
        for i in range(0, N):
            pressure[times[tin]][i] = np.sin(2 * np.pi * tin / T)
            velocity[times[tin]][i] = np.cos(2 * np.pi * tin / T)

    return pressure, velocity

def augment_time(field, period, ntimepoints):
    times_before = [t for t in field]
    times_before.sort()
    ntimes = len(times_before)

    npoints = field[times_before[0]].shape[0]

    times_scaled = np.linspace(0, period, ntimes)
    times_new = np.linspace(0, period, ntimepoints)

    Y = np.zeros((npoints, ntimepoints))
    for ipoint in range(npoints):
        y = []
        for t in times_before:
            y.append(field[t][ipoint])

        tck = interpolate.splrep(times_scaled, y, s=0)
        Y[ipoint,:] = interpolate.splev(times_new, tck, der=0)

    newfield = {}
    count = 0
    for t in times_new:
        newfield[t] = np.expand_dims(Y[:,count],axis=1)
        count = count + 1

    return newfield

def save_animation(pressure, velocity, filename):
    def find_min_max(field):
        times = [t for t in field]

        minv = np.infty
        maxv = np.NINF

        for t in times:
            curmin = np.min(field[t])
            curmax = np.max(field[t])
            if curmin < minv:
                minv = curmin
            if curmax > maxv:
                maxv = curmax
        return minv, maxv

    times = [t for t in pressure]
    minp, maxp = find_min_max(pressure)
    minv, maxv = find_min_max(velocity)

    fig, ax = plt.subplots(2)
    line_p, = ax[0].plot([],[],'r')
    line_v, = ax[1].plot([],[],'r')

    def animation_frame(i):
        line_p.set_xdata(range(0,len(pressure[times[i]])))
        line_p.set_ydata(pressure[times[i]])
        line_v.set_xdata(range(0,len(velocity[times[i]])))
        line_v.set_ydata(velocity[times[i]])
        ax[0].set_xlim(0,len(pressure[times[i]]))
        ax[0].set_ylim(minp-np.abs(minp)*0.1,maxp+np.abs(maxp)*0.1)
        ax[1].set_xlim(0,len(velocity[times[i]]))
        ax[1].set_ylim(minv-np.abs(minv)*0.1,maxv+np.abs(maxv)*0.1)
        ax[0].set_title('pressure ' + str(times[i]))
        ax[1].set_title('velocity ' + str(times[i]))
        return line_p, line_v

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(pressure),
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=60)
    anim.save(filename + '.mp4', writer = writervideo)

def generate_graphs(model_name, input_dir, save = True):
    geo, fields = create_geometry(model_name, input_dir, 15, remove_caps = True,
                                  points_to_keep = None, doresample = True)
    pressure, velocity = io.gather_pressures_velocities(fields)
    pressure, velocity, area = geo.generate_fields(pressure,
                                                   velocity,
                                                   fields['area'])

    # the period
    T = 0.7
    npoints = 15000

    print('Augmenting timesteps')
    pressure = augment_time(pressure, T, npoints)
    velocity = augment_time(velocity, T, npoints)
    # save_animation(pressure, velocity, 'interpolated_fields')

    print('Generating graphs')
    fixed_graph = create_fixed_graph(geo, area)

    print('Adding fields')
    graphs = add_fields(fixed_graph, pressure, velocity)
    if save:
        dgl.save_graphs('data/' + sys.argv[2], graphs)
    return graphs

if __name__ == "__main__":
    input_dir = 'vtps'
    generate_graphs(sys.argv[1], input_dir, True)
