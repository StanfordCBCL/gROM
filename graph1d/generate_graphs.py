# Copyright 2023 Stanford University

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

import sys
import os
sys.path.append(os.getcwd())
import tools.io_utils as io
import numpy as np
import scipy
from scipy import interpolate
import dgl
import torch as th
from tqdm import tqdm
import json
import random
import tools.plot_tools as pt
import matplotlib.pyplot as plt
import shutil

def generate_types(bif_id, indices):
    """
    Generate node types.

    Generate one-hot representation of node type: 0 = branch node, 1 = junction
    node, 2 = inlet, 3 = outlet.

    Arguments:
        bif_id: numpy array containing node types as read from .vtp
        indices: dictionary containing inlet and outlets indices
    Returns:
        One-hot representation of the node type
        Inlet mask, i.e., array containing 1 at inlet index and 0 elsewhere
        Outlet maks, i.e., array containing 1 at outlet indices and 0 elsewhere

    """
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
    types = th.nn.functional.one_hot(th.tensor(types), num_classes = 4)
    return types, inlet_mask, outlet_mask

def generate_edge_features(points, edges1, edges2):
    """
    Generate edge features.

    Returns a n x 3 array where row i contains (x_j - x_i) / |x_j - x_i|
    (node coordinates) and n is the number of nodes.
    Here, j and i are the node indices contained in row i of the edges1 and
    edges2 inputs. The second output is |x_j - x_i|.

    Arguments:
        points: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
    Returns:
        n x 3 numpy array containing x_j - x_i
        n dimensional numpy array containing |x_j - x_i|

    """
    rel_position = []
    rel_position_norm = []
    nedges = len(edges1)
    for i in range(nedges):
        diff = points[edges2[i],:] - points[edges1[i],:]
        ndiff = np.linalg.norm(diff)
        rel_position.append(diff / ndiff)
        rel_position_norm.append(ndiff)
    return np.array(rel_position), rel_position_norm

def add_fields(graph, field, field_name, offset = 0,
               pad = 10):
    """
    Add time-dependent fields to a DGL graph.

    Add time-dependent scalar fields as graph node features. The time-dependent
    fields are stored as n x 1 x m Pytorch tensors, where n is the number of
    graph nodes and m the number of timesteps.

    Arguments:
        graph: DGL graph
        field: dictionary with (key: timestep, value: field value)
        field_name (string): name of the field
        offset (int): number of timesteps to skip.
                      Default: 0 -> keep all timesteps
        pad (int): number of timesteps to add for interpolation from zero
                   zero initial conditions. Default: 0 -> start from actual
                   initial condition
    """
    timesteps = [float(t) for t in field]
    timesteps.sort()
    dt = (timesteps[1] - timesteps[0])
    T = timesteps[-1]
    # we use the third dimension for time
    field_t = th.zeros((list(field.values())[0].shape[0], 1,
                        len(timesteps) - offset + pad))

    times = [t for t in field]
    times.sort()
    times = times[offset:]

    loading_t = th.zeros((list(field.values())[0].shape[0], 1,
                          len(timesteps) - offset + pad), dtype = th.bool)

    if pad > 0:
        # def interpolate_function(count):
        #     return (1 - np.cos(np.pi * count / pad)) / 2

        inc = th.tensor(field[times[0]], dtype = th.float32)
        deft = inc * 0
        if field_name == 'pressure':

            minp = np.infty
            for t in field:
                minp = np.min((minp, np.min(field[t])))
            deft = deft + minp
        for i in range(pad):
            field_t[:,0,i] = deft * (pad - i)/pad + inc * (i / pad)
            loading_t[:,0,i] = True
    
    for i, t in enumerate(times):
        f = th.tensor(field[t], dtype = th.float32)
        field_t[:,0,i + pad] = f
        loading_t[:,0,i + pad] = False
        # graph.ndata[field_name + '_{}'.format(count - offset)] = f

    graph.ndata[field_name] = field_t
    graph.ndata['loading'] = loading_t
    graph.ndata['dt'] = th.reshape(th.ones(graph.num_nodes(),
                                   dtype = th.float32) * dt, (-1,1,1))
    graph.ndata['T'] = th.reshape(th.ones(graph.num_nodes(),
                                   dtype = th.float32) * T, (-1,1,1))

def find_outlets(edges1, edges2):
    """
    Find outlets.

    Find outlet indices given edge node indices.

    Arguments:
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge

    """
    outlets = []
    for e in edges2:
        if e not in edges1:
            outlets.append(e)
    return outlets

def remove_points(idxs_to_delete, idxs_to_replace, edges1, edges2, npoints):
    """
    Remove points.

    Remove points given their indices. This function is useful to find new
    connectivity arrays edges1 and edges2 after deleting nodes.

    Arguments:
        idxs_to_delete: indices of nodes to delete
        idxs_to_replace: indices of nodes that replace the deleted nodes.
                         Must have the same number of components as
                         idxs_to_delete
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        npoints: total number of nodes in the graph

    Returns:
        numpy array with indices of the remaining nodes
        (modified) numpy array containing indices of source nodes for every edge
        (modified) numpy array containing indices of dest nodes for every edge

    """
    npoints_to_delete = len(idxs_to_delete)

    for i in range(npoints_to_delete):
        i1 = np.where(edges1 == idxs_to_delete[i])[0]
        if (len(i1)) != 0:
            edges1[i1] = idxs_to_replace[i]

        i2 = np.where(edges2 == idxs_to_delete[i])[0]
        if (len(i2)) != 0:
            edges2[i2] = idxs_to_replace[i]

    edges_to_delete = np.where(edges1 == edges2)[0]
    edges1 = np.delete(edges1, edges_to_delete)
    edges2 = np.delete(edges2, edges_to_delete)

    sampled_indices = np.delete(np.arange(npoints), idxs_to_delete)
    for i in range(edges1.size):
        edges1[i] = np.where(sampled_indices == edges1[i])[0][0]
        edges2[i] = np.where(sampled_indices == edges2[i])[0][0]

    return sampled_indices, edges1, edges2

def resample_points(points, edges1, edges2, indices, perc_points_to_keep,
                    remove_caps):
    """
    Resample points.

    Select a subset of the points originally contained in the centerline.
    Specifically, this function retains perc_points_to_keep% points deleting
    those corresponding to the smallest edge sizes.

    Arguments:
        points: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        indices: dictionary containing inlet and outlets indices
        perc_points_to_keep (float): percentage of points to keep (in decimals)
        remove_caps (int): number of points to remove at the caps

    Returns:
        numpy array with indices of the remaining nodes
        (modified) n x 3 numpy array of point coordinates
        (modified) numpy array containing indices of source nodes for every edge
        (modified) numpy array containing indices of dest nodes for every edge
        (modified) dictionary containing inlet and outlets indices

    """

    def modify_edges(edges1, edges2, ipoint_to_delete, ipoint_to_replace):
        i1 = np.where(edges1 == ipoint_to_delete)[0]
        if len(i1) != 0:
            edges1[i1] = ipoint_to_replace

        i2 = np.where(np.array(edges2) == ipoint_to_delete)[0]
        if len(i2) != 0:
            edges2[i2] = ipoint_to_replace
        return edges1, edges2
    npoints = points.shape[0]
    npoints_to_keep = int(npoints * perc_points_to_keep)
    ipoints_to_delete = []
    ipoints_to_replace = []

    new_outlets = []
    for ip in range(remove_caps):
        for inlet in indices['inlet']:
            ipoints_to_delete.append(inlet + ip)
            ipoints_to_replace.append(inlet + remove_caps)
            edges1, edges2 = modify_edges(edges1, edges2,
                                          inlet + ip, inlet + remove_caps)
        for outlet in indices['outlets']:
            ipoints_to_delete.append(outlet - ip)
            ipoints_to_replace.append(outlet - remove_caps)
            edges1, edges2 = modify_edges(edges1, edges2,
                                          outlet - ip, outlet - remove_caps)

    for outlet in indices['outlets']:
        new_outlets.append(outlet - remove_caps)

    indices['outlets'] = new_outlets

    for _ in range(npoints - npoints_to_keep):
        diff = np.linalg.norm(points[edges1,:] - points[edges2,:],
                              axis = 1)
        # we don't consider the points that we already deleted
        diff[np.where(diff < 1e-13)[0]] = np.inf
        mdiff = np.min(diff)
        mind = np.where(np.abs(diff - mdiff) < 1e-12)[0][0]

        if edges2[mind] not in new_outlets:
            ipoint_to_delete = edges2[mind]
            ipoint_to_replace = edges1[mind]
        else:
            ipoint_to_delete = edges1[mind]
            ipoint_to_replace = edges2[mind]

        edges1, edges2 = modify_edges(edges1, edges2,
                                      ipoint_to_delete, ipoint_to_replace)

        ipoints_to_delete.append(ipoint_to_delete)
        ipoints_to_replace.append(ipoint_to_replace)

    sampled_indices, edges1, edges2 = remove_points(ipoints_to_delete,
                                                    ipoints_to_replace,
                                                    edges1, edges2,
                                                    npoints)

    points = np.delete(points, ipoints_to_delete, axis = 0)

    return sampled_indices, points, edges1, edges2, indices

def dijkstra_algorithm(nodes, edges1, edges2, index):
    """
    Dijkstra's algorithm.

    The algorithm finds the shortest paths from one node to every other node
    in the graph

    Arguments:
        nodes: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        index (int): index of the seed node

    Returns:
        numpy array with n components (n being the total number of nodes)
            containing all shortest path lengths
        numpy array with n components containing the previous nodes explored
            when traversing the graph

    """
    nnodes = nodes.shape[0]
    tovisit = np.arange(0,nnodes)
    dists = np.ones((nnodes)) * np.infty
    prevs = np.ones((nnodes)) * (-1)
    b_edges = np.array([edges1,edges2]).transpose()

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
                alt = dists[curindex] + np.linalg.norm(nodes[curindex,:] - \
                        nodes[neib,:])
                if alt < dists[neib]:
                    dists[neib] = alt
                    prevs[neib] = curindex
    if np.max(dists) == np.infty:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], s = 0.5, c = 'black')
        idx = np.where(dists > 1e30)[0]
        ax.scatter(nodes[idx,0], nodes[idx,1], nodes[idx,2], c = 'red')
        plt.show()
        raise ValueError("Distance in Dijkstra is infinite for some reason. You can try to adjust resample parameters.")
    return dists, prevs

def generate_boundary_edges(points, indices, edges1, edges2):
    """
    Generate boundary edges.

    Generate edges connecting boundary nodes to interior nodes. Every interior
    node is connected to the closest boundary node (in terms of path length).

    Arguments:
        points: n x 3 numpy array of point coordinates
        indices: dictionary containing inlet and outlets indices
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge

    Returns:
        numpy array containing indices of source nodes for every boundary edge
        numpy array containing indices of dest nodes for every boundary edge
        n x 3 numpy array containing (x_j - x_i) / |x_j - x_i|
        n dimensional numpy array containing, for every node, its distance to
            the closest boundary node (in terms of path length)

    """
    npoints = points.shape[0]
    idxs = indices['inlet'] + indices['outlets']
    bedges1 = []
    bedges2 = []
    rel_positions = []
    dists = []
    types = []
    for index in idxs:
        d, _ = dijkstra_algorithm(points, edges1, edges2, index)
        if index in indices['inlet']:
            type = 2
        else:
            type = 3
        for ipoint in range(npoints):
            bedges1.append(index)
            bedges2.append(ipoint)
            rp = points[ipoint,:] - points[index,:]
            rel_positions.append(rp)
            if np.linalg.norm(rp) > 1e-12:
                rel_positions[-1] = rel_positions[-1] / np.linalg.norm(rp)
            dists.append(d[ipoint])
            types.append(type)

        # DEBUG: check distances computed with dijkstra_algorithm
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # for i in range(npoints):
        #     ax.scatter(points[i,0], points[i,1], points[i,2], color = 'black', s = 1)
        #     ax.text(points[i,0], points[i,1], points[i,2],
        #             "{:.1f}".format(d[i]),
        #             fontsize=5)
        #     ax.set_box_aspect((np.ptp(points[:,0]),
        #                np.ptp(points[:,1]),
        #                np.ptp(points[:,2])))
        # plt.show()

    # we only keep edges corresponding to the closest boundary node in graph
    # distance to reduce number of edges
    edges_to_delete = []

    for ipoint in range(npoints):
        cur_dists = dists[ipoint::npoints]
        min_dist = np.min(cur_dists)
        minidx = np.where(np.abs(cur_dists - min_dist) < 1e-12)[0][0]
        if min_dist < 1e-12:
            edges_to_delete.append(ipoint + minidx * npoints)
        i = ipoint
        while i < len(dists):
            if i != ipoint + minidx * npoints:
                edges_to_delete.append(i)
            i = i + npoints

    bedges1 = np.delete(np.array(bedges1), edges_to_delete)
    bedges2 = np.delete(np.array(bedges2), edges_to_delete)
    rel_positions = np.delete(np.array(rel_positions), edges_to_delete,
                              axis = 0)
    dists = np.delete(np.array(dists), edges_to_delete)
    types = np.delete(np.array(types), edges_to_delete)

    # make edges bidirectional
    bedges1_copy = bedges1.copy()
    bedges1 = np.concatenate((bedges1, bedges2), axis = 0)
    bedges2 = np.concatenate((bedges2, bedges1_copy), axis = 0)
    rel_positions = np.concatenate((rel_positions, -rel_positions), axis = 0)
    dists = np.concatenate((dists, dists))
    types = np.concatenate((types, types))

    # DEBUG: plot all edges
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(points[:,0], points[:,1], points[:,2], color = 'black', s = 1)
    # for iedge in range(bedges1.size):
    #         ax.plot3D([points[bedges1[iedge],0],points[bedges2[iedge],0]],
    #                 [points[bedges1[iedge],1],points[bedges2[iedge],1]],
    #                 [points[bedges1[iedge],2],points[bedges2[iedge],2]],
    #                 color = 'black', linewidth=0.3, alpha = 0.5)
    # ax.set_box_aspect((np.ptp(points[:,0]),
    #             np.ptp(points[:,1]),
    #             np.ptp(points[:,2])))
    # plt.show()
    return bedges1, bedges2, rel_positions, dists, list(types)

def create_continuity_mask(types):
    """
    Generate mask to use when computing mass loss.

    Returns a numpy array containing 1s only at junction inlet indices.

    Arguments:
        types: n x m array containing the one-hot representation of node types

    Returns:
        n-dimensional numpy array containing 1s only at junction inlet indices

    """
    continuity_mask = [0]
    npoints = types.shape[0]
    for i in range(1,npoints-1):
        if types[i-1,0] == 1 and types[i,0] == 1 and types[i + 1,0]:
            continuity_mask.append(1)
        else:
            continuity_mask.append(0)
    continuity_mask.append(0)
    return continuity_mask

def create_junction_edges(points, bif_id, edges1, edges2, outlets):
    """
    Generate junction edges.

    Junction edges are bidirectional edges connecting junction inlets to
    the corresponding outlets.

    Arguments:
        points: n x 3 numpy array of point coordinates
        bif_id: n-dimensional array containing bifurcation (junction) ids
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        outlets: list of outlets indices

    Returns:
        numpy array containing indices of source nodes for every junction edge
        numpy array containing indices of dest nodes for every junction edge
        n x 3 numpy array containing (x_j - x_i) / |x_j - x_i|
        n dimensional numpy array containing, for every node, its distance to
            the closest boundary node (in terms of path length)
        list of n elements containing edge types (4)
        dictionary containing masks for inlet nodes and inlet+outlet nodes

    """
    npoints = bif_id.size
    jun_inlet_mask = [0] * npoints
    jun_mask = [0] * npoints
    juncts_inlets = {}
    jedges1 = []
    jedges2 = []
    for ipoint in range(npoints - 1):
        if ((bif_id[ipoint] == -1 and bif_id[ipoint + 1] != -1) or \
           (ipoint == 0 and bif_id[ipoint] != -1)) and \
           ipoint not in outlets:
            # we use the junction id as key and the junction idx as value
            if juncts_inlets.get(bif_id[ipoint + 1]) == None:
                juncts_inlets[bif_id[ipoint + 1]] = ipoint
                jun_inlet_mask[ipoint] = 1
                jun_mask[ipoint] = 1
        # we need to handle this case because sometimes -1 points disappear
        # between junctions when resampling
        elif bif_id[ipoint] != -1 and bif_id[ipoint - 1] != -1 and \
           bif_id[ipoint - 1] != bif_id[ipoint]:
            juncts_inlets[bif_id[ipoint]] = juncts_inlets[bif_id[ipoint-1]]
        elif bif_id[ipoint] == -1 and bif_id[ipoint - 1] != -1:
            # we look for the right inlet
            jedges1.append(juncts_inlets[bif_id[ipoint - 1]])
            jedges2.append(ipoint)
            jun_mask[ipoint] = 1
    masks = {'inlets': jun_inlet_mask, 'all': jun_mask}
    dists = {}
    for jun_id in juncts_inlets:
        d, _ = dijkstra_algorithm(points, edges1, edges2, juncts_inlets[jun_id])
        dists[juncts_inlets[jun_id]] = d

    jrel_position = []
    jdistance = []
    for iedg in range(len(jedges1)):
        jrel_position.append(points[jedges2[iedg],:] - points[jedges1[iedg],:])
        jdistance.append(dists[jedges1[iedg]][jedges2[iedg]])

    jrel_position = np.array(jrel_position)
    jdistance = np.array(jdistance)

    # make edges bidirectional
    jedges1_copy = jedges1.copy()
    jedges1 = jedges1 + jedges2
    jedges2 = jedges2 + jedges1_copy
    jrel_position = np.concatenate((jrel_position, -jrel_position), axis = 0)
    jdistance = np.concatenate((jdistance, jdistance))
    types = [4] * len(jedges1)
    return jedges1, jedges2, jrel_position, jdistance, types, masks

def load_vtp(file, input_dir):
    """
    Load vtp file.

    Arguments:
        file (string): file name
        input_dir (string): path to input_dir

    Returns:
        dictionary containing point data (key: name, value: data)
        n x 3 numpy array of point coordinates
        numpy array containing indices of source nodes for every edge
        numpy array containing indices of dest nodes for every edge

    """
    soln = io.read_geo(input_dir + '/' + file)
    point_data, _, points = io.get_all_arrays(soln.GetOutput())
    edges1, edges2 = io.get_edges(soln.GetOutput())

    # lets check for nans and delete points if they appear
    ni = np.argwhere(np.isnan(point_data['area']))
    if ni.size > 0:
        for i in ni[0]:
            indices = np.where(edges1 >= i)[0]
            edges1[indices] = edges1[indices] - 1

            indices = np.where(edges2 >= i)[0]
            edges2[indices] = edges2[indices] - 1

            indices = np.where(edges1 == edges2)[0]
            edges1 = np.delete(edges1,indices)
            edges2 = np.delete(edges2,indices)

            points = np.delete(points, i, axis = 0)
            for ndata in point_data:
                point_data[ndata] = np.delete(point_data[ndata], i)

    return point_data, points, edges1, edges2

def generate_tangents(points, branch_id):
    """
    Generate tangents.

    Generate tangent vector at every graph node.

    Arguments:
        points: n x 3 numpy array of point coordinates
        branch_id: n-dimensional array containing branch ids

    Returns:
        n x 3 numpy array of normalized tangent vectors

    """
    tangents = np.zeros(points.shape)
    maxbid = int(np.max(branch_id))
    for bid in range(maxbid + 1):
        point_idxs = np.where(branch_id == bid)[0]

        tck, u = scipy.interpolate.splprep([points[point_idxs, 0],
                                            points[point_idxs, 1],
                                            points[point_idxs, 2]], s=0,
                                            k = np.min((3, len(point_idxs)-1)))
        

        x, y, z = interpolate.splev(u, tck, der = 1)
        tangents[point_idxs,0] = x
        tangents[point_idxs,1] = y
        tangents[point_idxs,2] = z

    # make sure tangents are unitary
    tangents = tangents / np.linalg.norm(tangents, axis = 0)

    for i in range(tangents.shape[0]):
        tangents[i] = tangents[i] / np.linalg.norm(tangents[i])

    return tangents

def generate_graph(point_data, points, edges1, edges2,
                   add_boundary_edges, add_junction_edges,
                   rcr_values):
    """
    Generate graph.

    Generate DGL graph out of data obtained from a vtp file.

    Arguments:
        point_data: dictionary containing point data (key: name, value: data)
        points: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        add_boundary_edges (bool): decide whether to add boundary edges
        add_junction_edges (bool): decide whether to add junction edges
        rcr_values: dictionary associating each branch id outlet to values 
                    of RCR boundary conditions

    Returns:
        DGL graph
        dictionary containing indices of inlet and outlet nodes
        n x 3 numpy array of point coordinates
        n-dimensional array containin junction ids
        numpy array containing indices of source nodes for every edge
        numpy array containing indices of dist nodes for every edge
    """

    inlet = [0]
    outlets = find_outlets(edges1, edges2)

    indices = {'inlet': inlet,
               'outlets': outlets}

    bif_id = point_data['BifurcationId']

    try:
        area = list(io.gather_array(point_data, 'area').values())[0]
    except Exception:
        area = point_data['area']

    # we manually make the graph bidirected in order to have the relative
    # position of nodes make sense (xj - xi = - (xi - xj)). Otherwise, each edge
    # will have a single feature
    edges1_copy = edges1.copy()
    edges1 = np.concatenate((edges1, edges2))
    edges2 = np.concatenate((edges2, edges1_copy))

    rel_position, distance = generate_edge_features(points, edges1, edges2)

    types, inlet_mask, \
    outlet_mask = generate_types(bif_id, indices)

    # we need to find the closest point in the rcr file, because the
    # id might be different if we used different centerlines for 
    # solution and generation of the rcr file
    def find_closest_point_in_rcr_file(point):
        min_d = np.infty
        sid = -1
        for id in rcr_values:
            if type(rcr_values[id]) is dict and 'point' in rcr_values[id]:
                diff = np.linalg.norm(point - np.array(rcr_values[id]['point']))
                if diff < min_d:
                    min_d = diff
                    sid = id
        return sid
    npoints = points.shape[0]
    rcr = np.zeros((npoints,3))
    for ipoint in range(npoints):
        if outlet_mask[ipoint] == 1:
            if rcr_values['bc_type'] == 'RCR':
                id = find_closest_point_in_rcr_file(points[ipoint])
                rcr[ipoint,:] = rcr_values[id]['RCR']
            elif rcr_values['bc_type'] == 'R':
                id = find_closest_point_in_rcr_file(points[ipoint])
                rcr[ipoint,0] = rcr_values[id]['RP'][0]
            else:
                raise ValueError('Unknown type of boundary conditions!')
    etypes = [0] * edges1.size
    # we set etype to 1 if either of the nodes is a junction
    for iedge in range(edges1.size):
        if types[edges1[iedge],1] == 1 or types[edges2[iedge],1] == 1:
            etypes[iedge] = 1

    if add_boundary_edges:
        bedges1, bedges2, \
        brel_position, bdistance, \
        btypes = generate_boundary_edges(points, indices, edges1, edges2)
        edges1 = np.concatenate((edges1, bedges1))
        edges2 = np.concatenate((edges2, bedges2))
        etypes = etypes + btypes
        distance = np.concatenate((distance, bdistance))
        rel_position = np.concatenate((rel_position, brel_position), axis = 0)

    if add_junction_edges and np.max(bif_id) > -1:
        jedges1, jedges2, \
        jrel_position, jdistance, \
        jtypes, jmasks = create_junction_edges(points, bif_id,
                                               edges1,
                                               edges2,
                                               outlets)
        edges1 = np.concatenate((edges1, jedges1))
        edges2 = np.concatenate((edges2, jedges2))
        etypes = etypes + jtypes
        distance = np.concatenate((distance, jdistance))
        rel_position = np.concatenate((rel_position, jrel_position), axis = 0)
    else:
        jmasks = {}
        jmasks['inlets'] = np.zeros(bif_id.size)
        jmasks['all'] = np.zeros(bif_id.size)

    graph = dgl.graph((edges1, edges2), idtype = th.int32)

    graph.ndata['x'] = th.tensor(points, dtype = th.float32)
    tangent = th.tensor(point_data['tangent'], dtype = th.float32)
    graph.ndata['tangent'] = th.unsqueeze(tangent, 2)
    graph.ndata['area'] = th.reshape(th.tensor(area, dtype = th.float32),
                                     (-1,1,1))
    continuity_mask = create_continuity_mask(types)

    graph.ndata['type'] = th.unsqueeze(types, 2)
    graph.ndata['inlet_mask'] = th.tensor(inlet_mask, dtype = th.int8)
    graph.ndata['outlet_mask'] = th.tensor(outlet_mask, dtype = th.int8)
    graph.ndata['continuity_mask'] = th.tensor(continuity_mask, dtype = th.int8)
    graph.ndata['jun_inlet_mask'] = th.tensor(jmasks['inlets'], dtype = th.int8)
    graph.ndata['jun_mask'] = th.tensor(jmasks['all'], dtype = th.int8)
    graph.ndata['branch_mask'] = th.tensor(types[:,0].detach().numpy() == 1,
                                           dtype = th.int8)
    graph.ndata['branch_id'] = th.tensor(point_data['BranchId'],
                                         dtype = th.int8)

    graph.ndata['resistance1'] = th.reshape(th.tensor(rcr[:,0], dtype=th.float32), (-1,1,1))
    graph.ndata['capacitance'] = th.reshape(th.tensor(rcr[:,1], dtype=th.float32), (-1,1,1))
    graph.ndata['resistance2'] = th.reshape(th.tensor(rcr[:,2], dtype=th.float32), (-1,1,1))

    graph.edata['rel_position'] = th.unsqueeze(th.tensor(rel_position,
                                               dtype = th.float32), 2)
    graph.edata['distance'] = th.reshape(th.tensor(distance,
                                         dtype = th.float32), (-1,1,1))
    etypes = th.nn.functional.one_hot(th.tensor(etypes), num_classes = 5)
    graph.edata['type'] = th.unsqueeze(etypes, 2)

    return graph, indices, points, bif_id, edges1, edges2

def create_partitions(points, bif_id,
                      edges1, edges2, max_num_partitions):
    """
    Generate partitions out of a graph.

    Generate partitions out of a graph. This function splits the graph into
    multiple subgraphs, making sure that junctions are kept in the same
    partition.

    Arguments:
        points: n x 3 numpy array of point coordinates
        bif_id: n-dimensional array containin junction ids
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        max_num_partitions (int): maximum number of partitions

    Returns:
        list of partitions
    """

    def create_partition(edges1, edges2, starting_point, inlets):
        sampling_indices = [starting_point]
        new_edges1 = []
        new_edges2 = []
        points_to_visit = [starting_point]
        count = 0
        numbering = {starting_point: count}
        count = count + 1
        while len(points_to_visit) > 0:
            j = points_to_visit[0]
            del points_to_visit[0]
            iedges = np.where(edges1 == j)[0]
            for iedg in iedges:
                next_point = edges2[iedg]
                numbering[next_point] = count
                count = count + 1
                sampling_indices.append(next_point)
                new_edges1.append(numbering[j])
                new_edges2.append(numbering[next_point])
                if next_point not in inlets:
                    points_to_visit.append(next_point)

        return np.array(new_edges1), np.array(new_edges2), sampling_indices

    bif_id = point_data['BifurcationId']
    npoints = bif_id.size

    inlets = [0]
    # num_partions is the number of inlets that we have to randomly select from
    # the graph. So we start by randoming selecting one inlet between each
    # couple of consecutive bifurcations, and then we randomly select the
    # following inlets
    for ipoint in range(npoints):
        if len(inlets) == max_num_partitions:
            break
        # then it's the outlet of a junction, we traverse the graph and until we
        # can. If we reach an outlet, we do nothing. If we reach another
        # junction, we sample a point between these two indices
        if bif_id[ipoint] != -1 and bif_id[ipoint+1] == -1:
            j = ipoint
            next = -1
            while True:
                iedg = np.where(edges1 == j)[0]
                if len(iedg) == 0:
                    break
                j = edges2[iedg[0]]
                if bif_id[j] != -1:
                    next = j
                    break
            if next != -1:
                inlets.append(int(np.random.randint(ipoint +1 , next)))

    if len(inlets) < max_num_partitions:
        available_in = list(np.where(bif_id == -1)[0])
        # we allow for a max of 2 straight partitions
        n_new_inlets = np.min((max_num_partitions - len(inlets), 2))
        inlets = inlets + random.sample(available_in, n_new_inlets)

    partitions = []

    for ipartition in range(len(inlets)):
        pedges1, pedges2, sampling_indices = create_partition(edges1, edges2,
                                                            inlets[ipartition],
                                                            inlets)
        ppoints = points[sampling_indices,:]

        ppoint_data = {}
        for ndata in point_data:
            ppoint_data[ndata] = point_data[ndata][sampling_indices]

        new_partition = {'edges1': pedges1,
                         'edges2': pedges2,
                         'points': ppoints,
                         'sampling_indices': sampling_indices,
                         'point_data': ppoint_data}
        if pedges1.size > 1:
            partitions.append(new_partition)
    return partitions

def resample_time(field, timestep, period, shift = 0):
    """
    Resample timesteps.

    Given a time-dependent field distributed over graph nodes, this function
    resamples the field in time using B-spline interpolation at every node.

    Arguments:
        field: dictionary containing the field for all timesteps
               (key: timestep, value: n-dimensional numpy array)
        timestep (float): the new timestep
        period (float): period of the simulation. We restrict to one cardiac
                        cycle

        shift (float): apply shift (s) to start at the beginning of the systole.
                       Default value -> 0

    Returns:
        Dictionary containing the field for all resampled timesteps
            (key: timestep, value: n-dimensional numpy array)
    """
    original_timesteps = [t for t in field]
    original_timesteps.sort()

    t0 = original_timesteps[0]
    T = original_timesteps[-1]
    t = [t0 + shift]
    nnodes = field[t0].size
    resampled_field = {t0 + shift: np.zeros(nnodes)}
    while t[-1] < T and t[-1] <= t[0] + period:
        t.append(t[-1] + timestep)
        resampled_field[t[-1]] = np.zeros(nnodes)

    for inode in range(nnodes):
        values = []
        for time in original_timesteps:
            values.append(field[time][inode])

        tck, _ = scipy.interpolate.splprep([values],
                                           u = original_timesteps, s = 0)
        values_interpolated = interpolate.splev(t, tck)[0]

        for i, time in enumerate(t):
            resampled_field[time][inode] = values_interpolated[i]

    return resampled_field

"""
The main function reads all vtps files from the folder specified in input_dir
and generates DGL graphs. The graphs are saved in output_dir.
"""
if __name__ == "__main__":
    data_location = io.data_location()
    input_dir = data_location + 'vtps/'
    output_dir = data_location + 'graphs/'

    dataset_info = json.load(open(input_dir + '/dataset_info.json'))

    files = os.listdir(input_dir)

    print('Processing all files in {}'.format(input_dir))
    print('File list:')
    print(files)
    for file in tqdm(files, desc = 'Generating graphs', colour='green'):
        if '.vtp' in file and 's' in file:
            point_data, points, edges1, edges2 = load_vtp(file, input_dir)
            try:
                point_data['tangent'] = generate_tangents(points,
                                                      point_data['BranchIdTmp'])
            except Exception:
                continue
            inlet = [0]
            outlets = find_outlets(edges1, edges2)

            indices = {'inlet': inlet,
                       'outlets': outlets}

            resample_perc = 0.06
            success = False

            while not success:
                try:
                    sampled_indices, points, \
                    edges1, edges2, _ = resample_points(points.copy(),
                                                    edges1.copy(),
                                                    edges2.copy(), indices,
                                                    resample_perc,
                                                    remove_caps = 3)
                    success = True
                except Exception as e:
                    print(e)
                    resample_perc = np.min([resample_perc * 2, 1])

            for ndata in point_data:
                point_data[ndata] = point_data[ndata][sampled_indices]

            inlet = [0]
            outlets = find_outlets(edges1, edges2)

            indices = {'inlet': inlet,
                       'outlets': outlets}

            pressure = io.gather_array(point_data, 'pressure')
            flowrate = io.gather_array(point_data, 'flow')
            if len(flowrate) == 0:
                flowrate = io.gather_array(point_data, 'velocity')

            times = [t for t in pressure]
            timestep = float(dataset_info[file.replace('.vtp','')]['dt'])
            for t in times:
                pressure[t * timestep] = pressure[t]
                flowrate[t * timestep] = flowrate[t]
                del pressure[t]
                del flowrate[t]

            # scale pressure to be mmHg
            for t in pressure:
                pressure[t] = pressure[t] / 1333.2

            times = [t for t in pressure]

            sampling_inqdices = np.arange(points.shape[0])
            part = {'point_data': point_data,
                     'points': points,
                     'edges1': edges1,
                     'edges2': edges2,
                     'sampling_indices': sampling_indices}

            add_boundary_edges = True
            add_junction_edges = False

            fname = file.replace('.vtp','')
            graph, indices, \
            points, bif_id, \
            edges1, edges2 = generate_graph(part['point_data'],
                                            part['points'],
                                            part['edges1'],
                                            part['edges2'],
                                            add_boundary_edges,
                                            add_junction_edges,
                                            dataset_info[fname])

            do_resample_time = True
            ncopies = 1
            if do_resample_time:
                ncopies = 4
                dt = 0.01
                offset = int(np.floor((dt / timestep) / ncopies))

            intime = 0
            for icopy in range(ncopies):
                c_pressure = {}
                c_flowrate = {}
                
                for t in times[intime:]:
                    c_pressure[t] = pressure[t][part['sampling_indices']]
                    c_flowrate[t] = flowrate[t][part['sampling_indices']]

                if do_resample_time:
                    period = dataset_info[fname]['T']
                    shift = dataset_info[fname]['time_shift']
                    c_pressure = resample_time(c_pressure, timestep = dt, 
                                            period = period,
                                            shift = shift)
                    c_flowrate = resample_time(c_flowrate, 
                                            timestep = dt,
                                            period =  period,
                                            shift = shift)
                    intime = intime + offset

                padt = 0.1
                add_fields(graph, c_pressure, 'pressure', pad = int(padt / dt))
                add_fields(graph, c_flowrate, 'flowrate', pad = int(padt / dt))

                filename = file.replace('.vtp','.' + str(icopy) + '.grph')
                dgl.save_graphs(output_dir + filename, graph)

    shutil.copy(input_dir + 'dataset_info.json',  
                output_dir + 'dataset_info.json')
