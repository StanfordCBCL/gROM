import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("../tools/")

import io_utils as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dgl
import torch as th
from tqdm import tqdm

def plot_graph(points, bif_id, indices):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax._axis3don = False

    minc = np.min(bif_id)
    maxc = np.max(bif_id)

    if minc == maxc:
        C = bif_id * 0
    else:
        C = (bif_id - minc) / (maxc - minc)

    cmap = cm.get_cmap("viridis")
    ax.scatter(points[:,0], points[:,1], points[:,2], color=cmap(C),            depthshade=0, s = 5)

    inlet = indices['inlet']
    ax.scatter(points[inlet,0], points[inlet,1], points[inlet,2],               color='green', depthshade=0, s = 60)

    outlets = indices['outlets']
    ax.scatter(points[outlets,0], points[outlets,1], points[outlets,2],color='red', depthshade=0, s = 60)

    # ax.set_xlim([points[outlets[0],0]-0.1,points[outlets[0],0]+0.1])
    # ax.set_ylim([points[outlets[0],1]-0.1,points[outlets[0],1]+0.1])
    # ax.set_zlim([points[outlets[0],2]-0.1,points[outlets[0],2]+0.1])

    plt.show()

def generate_types(bif_id, indices):
    types = []
    inlet_mask = []
    outlet_mask = []
    for i, id in enumerate(bif_id):
        if id == -1:
            cur_type = 0
        else:
            cur_type = 1
        if i in indices['inlet']:
            cur_type = 2
        elif i in indices['outlets']:
            cur_type = 3
        types.append(cur_type)
        if cur_type == 2:
            inlet_mask.append(True)
        else:
            inlet_mask.append(False)
        if cur_type == 3:
            outlet_mask.append(True)
        else:
            outlet_mask.append(False)
    return th.nn.functional.one_hot(th.tensor(types)), inlet_mask, outlet_mask

def generate_edge_features(points, edges1, edges2):
    rel_position = []
    rel_position_norm = []
    nedges = len(edges1)
    for i in range(nedges):
        diff = points[edges2[i],:] - points[edges1[i],:]
        ndiff = np.linalg.norm(diff)
        rel_position.append(diff / ndiff)
        rel_position_norm.append(ndiff)
    return np.array(rel_position), rel_position_norm

def add_fields(graph, field, field_name, subsample_time = 10):
    timesteps = [float(t) for t in field]
    timesteps.sort()
    dt = (timesteps[1] - timesteps[0]) * subsample_time
    # we skip the first 100 timesteps
    offset = 100
    count = 0
    # we use the third timension for time
    field_t = th.zeros((list(field.values())[0].shape[0], 1, 
                        len(timesteps) - offset))
    for t in field:
        if count >= offset:
            f = th.tensor(field[t], dtype = th.float32)
            field_t[:,0,count - offset] = f
            # graph.ndata[field_name + '_{}'.format(count - offset)] = f
        count = count + 1
    graph.ndata[field_name] = field_t[:,:,::subsample_time]
    graph.ndata['dt'] = th.reshape(th.ones(graph.num_nodes(), 
                                   dtype = th.float32) * dt, (-1,1,1))

def find_outlets(edges1, edges2):
    outlets = []
    for e in edges2:
        if e not in edges1:
            outlets.append(e)
    return outlets   

def resample_points(points, edges1, edges2, outlets, perc_points_to_keep):
    npoints = points.shape[0]
    npoints_to_keep = int(npoints * perc_points_to_keep)
    ipoints_to_delete = []
    for _ in range(npoints - npoints_to_keep):
        diff = np.linalg.norm(points[edges1,:] - points[edges2,:],
                              axis = 1)
        # we don't consider the points that we already deleted
        diff[np.where(diff < 1e-13)[0]] = np.inf
        mdiff = np.min(diff)
        mind = np.where(np.abs(diff - mdiff) < 1e-12)[0][0]

        if edges2[mind] not in outlets:
            ipoint_to_delete = edges2[mind]
            ipoint_to_replace = edges1[mind]
        else:
            ipoint_to_delete = edges1[mind]
            ipoint_to_replace = edges2[mind]

        i1 = np.where(edges1 == ipoint_to_delete)[0]
        if len(i1) != 0:   
            edges1[i1] = ipoint_to_replace
        
        i2 = np.where(np.array(edges2) == ipoint_to_delete)[0]
        if len(i2) != 0:
            edges2[i2] = ipoint_to_replace

        ipoints_to_delete.append(ipoint_to_delete)
    
    diff = np.linalg.norm(points[edges1,:] - points[edges2,:],
                              axis = 1)

    points = np.delete(points, ipoints_to_delete, axis = 0)

    edges_to_delete = np.where(diff < 1e-13)[0]
    edges1 = np.delete(edges1, edges_to_delete)
    edges2 = np.delete(edges2, edges_to_delete)

    sampled_indices = np.delete(np.arange(npoints), ipoints_to_delete)

    for i in range(edges1.size):
        edges1[i] = np.where(sampled_indices == edges1[i])[0][0]
        edges2[i] = np.where(sampled_indices == edges2[i])[0][0]

    return sampled_indices, points, edges1, edges2 
     
def generate_graph(file, input_dir, resample_perc):
    success = False
    soln = io.read_geo(input_dir + '/' + file)
    point_data, _, points = io.get_all_arrays(soln.GetOutput())
    edges1, edges2 = io.get_edges(soln.GetOutput())

    inlet = [0]
    outlets = find_outlets(edges1, edges2)

    indices = {'inlet': inlet,
               'outlets': outlets}

    sampled_indices, points, edges1, edges2 = resample_points(points, edges1, 
                                                              edges2, outlets,
                                                              resample_perc)

    count = 0
    for outlet in indices['outlets']:
        iout = int(np.where(sampled_indices == outlet)[0])
        indices['outlets'][count] = iout
        count = count + 1


    bif_id = point_data['BifurcationId'][sampled_indices]
    area = list(io.gather_array(point_data, 'area').values())[0]
    area = area[sampled_indices]

    # plot_graph(points, bif_id, indices)
    # we manually make the graph bidirected in order to have the relative 
    # position of nodes make sense (xj - xi = - (xi - xj)). Otherwise, each edge
    # will have a single feature
    edges1_copy = edges1.copy()
    edges1 = np.concatenate((edges1, edges2))
    edges2 = np.concatenate((edges2, edges1_copy))

    graph = dgl.graph((edges1, edges2), idtype = th.int32)

    graph.ndata['x'] = th.tensor(points, dtype = th.float32)
    graph.ndata['area'] = th.reshape(th.tensor(area, dtype = th.float32), 
                                     (-1,1,1))
    types, inlet_mask, \
    outlet_mask = generate_types(bif_id, indices)
    graph.ndata['type'] = th.unsqueeze(types, 2)

    graph.ndata['inlet_mask'] = th.tensor(inlet_mask, dtype = th.int8)
    graph.ndata['outlet_mask'] = th.tensor(outlet_mask, dtype = th.int8)

    rel_position, norm_rel_positions = generate_edge_features(points, 
                                                              edges1,
                                                              edges2)
    graph.edata['rel_position'] = th.unsqueeze(th.tensor(rel_position, 
                                               dtype = th.float32), 2)
    graph.edata['rel_position_norm'] = th.reshape(th.tensor(norm_rel_positions, 
                                                  dtype = th.float32), (-1,1,1))

    return graph, point_data, indices, sampled_indices

if __name__ == "__main__":
    data_location = io.data_location()
    input_dir = data_location + 'vtps_1D'
    output_dir = data_location + 'graphs/'

    files = os.listdir(input_dir)    

    print('Processing all files in {}'.format(input_dir))
    print('File list:')
    print(files)
    for file in tqdm(files, desc = 'Generating graphs', colour='green'):
        if '.vtp' in file:
            filename = file.replace('.vtp','.grph')
            
            resample_perc = 0.03
            success = False
            while not success:
                try:
                    graph, point_data, indices, \
                    sampled_indices = generate_graph(file, input_dir, 
                                                     resample_perc)
                    success = True
                except Exception as e:
                    resample_perc = resample_perc + 0.01
            
            pressure = io.gather_array(point_data, 'pressure')
            flowrate = io.gather_array(point_data, 'flow')

            # select indices and scale pressure to be mmHg
            for t in pressure:
                pressure[t] = pressure[t][sampled_indices]  / 1333.2
                flowrate[t] = flowrate[t][sampled_indices]

            add_fields(graph, pressure, 'pressure')
            add_fields(graph, flowrate, 'flowrate')

            dgl.save_graphs(output_dir + filename, graph)
