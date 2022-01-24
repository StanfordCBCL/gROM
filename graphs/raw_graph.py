import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt

class RawGraph:
    def __init__(self, points, params):
        self.points = points
        self.params = params

        self.divide()
        self.compute_h()
        self.find_junctions()
        self.resample()
        self.construct_interpolation_matrices()

    def divide(self):
        # we assume inlet is the first point
        self.inlet = 0
        npoints = self.points.shape[0]

        dists = np.linalg.norm(self.points[1:,:] - self.points[:-1,:], axis = 1)

        self.outlets = np.where(dists > self.params['tol'])[0]

        # we assume last point is an outlet
        self.outlets = np.append(self.outlets, self.points.shape[0]-1)
        self.portions = self.partition(self.points)

    def find_junctions(self):
        # there is one junction for each portion after 1 (the inlet)
        self.junctions = []
        nportions = len(self.portions)
        for rs_idx in range(1, nportions):
            mindists = []
            minidxs = []

            for other_idx in range(0, nportions):
                if rs_idx != other_idx:
                    diff = np.linalg.norm(self.portions[other_idx] - \
                                          self.portions[rs_idx][0, :], axis = 1)
                    mindist = np.min(diff)
                    minidx = np.where(np.abs(diff - mindist) < 1e-14)[0][0]
                else:
                    mindist = np.infty
                    minidx = -1

                mindists.append(mindist)
                minidxs.append(minidx)

            mindists = np.array(mindists)
            totalmin = np.min(mindists)
            portionidx = np.where(np.abs(mindists - totalmin) < 1e-14)[0][0]
            junction = {}
            junction['portion_1'] = portionidx
            junction['portion_2'] = rs_idx
            junction['point_1'] = minidxs[portionidx]
            junction['point_2'] = 0

            self.junctions.append(junction)

    def compute_portion_length(self, portion):
        return np.sum(np.linalg.norm(portion[1:,:] - portion[:-1,:], axis=1))

    def compute_h(self):
        tot_length = 0
        tot_points = 0
        for portion in self.portions:
            tot_points += portion.shape[0]
            tot_length += self.compute_portion_length(portion)

        self.h = tot_length / tot_points

    def partition(self, array):
        if len(array.shape) == 1:
            array = np.expand_dims(array, axis = 1)
        slices = []
        points_to_remove = 3
        prev = 0
        for outlet in self.outlets:
            offset_in = 0
            if self.params['remove_caps'] and prev == 0:
                offset_in = points_to_remove
            offset_out = 0
            if self.params['remove_caps']:
                offset_out = points_to_remove
            slices.append(array[prev + offset_in:outlet+1 - offset_out,:])
            prev = outlet+1

        return slices

    def set_node_types(self, bifurcation_id):
        bid = self.project(bifurcation_id)

        for i in range(len(bid)):
            bid[i] = np.round(bid[i]).astype(int)

        allids = set()
        for ids in bid:
            for id in ids:
                allids.add(int(id))

        degrees = {}
        count = 0
        for id in allids:
            degrees[id] = 0
            for bifid in bid:
                if id in np.squeeze(bifid).tolist():
                    degrees[id] = degrees[id] + 1

            count = count + 1

        degrees[-1] = 0

        for id in allids:
            for i in range(len(bid)):
                idxs = np.where(bid[i] == id)[0]
                bid[i][idxs] = degrees[id]

        # fix singular points
        for i in range(len(bid)):
            curbid = np.squeeze(bid[i])

            # find consecutive chunks
            idxs = np.where(curbid != 0)[0]


            i1 = [idxs[0]]
            i2 = []
            for j in range(len(idxs)-2):
                if idxs[j] + 1 != idxs[j+1]:
                    i2.append(idxs[j])
                    i1.append(idxs[j+1])

            i2.append(idxs[len(idxs)-1])

            # set chunk to most common value
            for chunki in range(len(i1)):
                cv = np.bincount(curbid[i1[chunki]:i2[chunki]+1]).argmax()
                bid[i][i1[chunki]:i2[chunki]+1] = cv

        self.nodes_type = bid

    def project(self, field):
        proj_field = []
        sliced_field = self.partition(field)

        for ifield in range(len(sliced_field)):
            weights = np.linalg.solve(self.interpolation_matrices[ifield],
                                      sliced_field[ifield])
            proj_values = np.matmul(self.projection_matrices[ifield], weights)

            proj_field.append(proj_values)

        return proj_field

    def resample(self):
        resampled_portions = []
        resampled_junctions = []
        resampled_tangents = []
        count = 0
        for portion in self.portions:
            # compute h of the portion
            alength = self.compute_portion_length(portion)
            N = int(np.floor(alength / (self.params['coarsening'] * self.h)))

            if N < 2:
                raise ValueError("Too few points in portion, \
                                  decrease resample frequency.")

            tck, u = scipy.interpolate.splprep([portion[:,0],
                                                portion[:,1],
                                                portion[:,2]], s=0, k = 3)
            u_fine = np.linspace(0, 1, N + 1)
            x, y, z = interpolate.splev(u_fine, tck)
            r_portion = np.vstack((x,y,z)).transpose()

            # we add the bifurcation point not at the extremum
            for junction in self.junctions:
                if junction['portion_1'] == count:
                    point = self.portions[count][junction['point_1'],:]
                    diff = np.linalg.norm(r_portion - point,axis = 1)
                    mindiff = np.min(diff)
                    minidx = np.where(np.abs(diff - mindiff) < 1e-12)[0][0]

                    if minidx == 0:
                        if diff[1] > np.linalg.norm(r_portion[1,:] - \
                                                    r_portion[0,:]):
                            r_portion = np.insert(r_portion, 0, point, axis = 0)
                            idx = 0
                        else:
                            r_portion = np.insert(r_portion, 1, point, axis = 0)
                            idx = 1
                    elif minidx == r_portion.shape[0]-1:
                        if diff[-1] > np.linalg.norm(r_portion[-1,:] - \
                                                     r_portion[-2,:]):
                            r_portion = np.append(r_portion, point, axis = 0)
                            idx = r_portion.shape[0]-1
                        else:
                            r_portion = np.insert(r_portion,
                                                  r_portion.shape[0]-1,
                                                  point, axis = 0)
                            idx = r_portion.shape[0]-2
                    else:
                        if diff[minidx+1] < diff[minidx-1]:
                            r_portion = np.insert(r_portion, minidx+1,
                                                  point, axis = 0)
                            idx = minidx+1
                        else:
                            r_portion = np.insert(r_portion, minidx,
                                                  point, axis = 0)
                            idx = minidx
                    r_junction = {}
                    r_junction['portion_1'] = count
                    r_junction['point_1'] = idx
                    r_junction['portion_2'] = junction['portion_2']
                    r_junction['point_2'] = junction['point_2']
                    resampled_junctions.append(r_junction)

                # we reinterpolate to get tangent after junction injection
                tck, u = scipy.interpolate.splprep([r_portion[:,0],
                                                    r_portion[:,1],
                                                    r_portion[:,2]], s=0, k = 3)

                x, y, z = interpolate.splev(u, tck, der = 1)
                tangent_portion = np.vstack((x,y,z)).transpose()
                tangent_portion = tangent_portion
                tnorms = np.linalg.norm(tangent_portion, axis = 1)
                for i in range(tnorms.size):
                    tangent_portion[i,:] = tangent_portion[i,:] / tnorms[i]

            resampled_portions.append(r_portion)
            resampled_tangents.append(tangent_portion)
            count = count + 1

        self.resampled_portions = resampled_portions
        self.resampled_junctions = resampled_junctions
        self.tangents = resampled_tangents

    def construct_interpolation_matrices(self):
        p_matrices = []
        i_matrices = []
        stdevcoeff = 40

        def kernel(nnorm, h):
            # 99% of the gaussian distribution is within 3 stdev from the mean
            return np.exp(-(nnorm / (2 * (h * stdevcoeff)**2)))

        for ipor in range(0,len(self.portions)):
            N = self.resampled_portions[ipor].shape[0]
            M = self.portions[ipor].shape[0]
            new_matrix = np.zeros((N,M))

            hs = []
            for j in range(0,M):
                h1 = -1
                h2 = -1
                if j != M-1:
                    h1 = np.linalg.norm(self.portions[ipor][j+1,:] - \
                                        self.portions[ipor][j,:])
                if j != 0:
                    h2 = np.linalg.norm(self.portions[ipor][j,:] - \
                                        self.portions[ipor][j-1,:])
                h = np.max((h1, h2))
                hs.append(h)
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(self.resampled_portions[ipor][i,:] -
                                       self.portions[ipor][j,:])
                    # we consider 4 stdev to be safe
                    # if n < 4 * hs[j] * stdevcoeff:
                    new_matrix[i,j] = kernel(n,hs[j])

            p_matrices.append(new_matrix)

            N = self.portions[ipor].shape[0]
            M = N
            new_matrix = np.zeros((N,M))
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(self.portions[ipor][i,:] - \
                                       self.portions[ipor][j,:])
                    # if n < 4 * hs[j] * stdevcoeff:
                    new_matrix[i,j] = kernel(n,hs[j])

            i_matrices.append(new_matrix)

        self.projection_matrices = p_matrices
        self.interpolation_matrices = i_matrices

    def stack(self, a_list):
        res = a_list[0]

        for subfield in a_list[1:]:
            res = np.concatenate((res, subfield), axis = 0)

        return res

    def partition_and_stack_field(self, field):
        p_field = self.project(field)
        res = self.stack(p_field)

        return res

    def create_homogeneous_graph(self):
        def seq_array(minv, maxv):
            return np.expand_dims(np.arange(minv,maxv+1), axis = 1)
        nodes = self.resampled_portions[0]
        edges = np.concatenate((seq_array(0,nodes.shape[0]-2), \
                                seq_array(1,nodes.shape[0]-1)), axis = 1)

        offsets = [0, nodes.shape[0]]
        for portion in self.resampled_portions[1:]:
            nodes = np.concatenate((nodes,portion), axis = 0)
            new_edges = np.concatenate((seq_array(0,portion.shape[0]-2), \
                                        seq_array(1,portion.shape[0]-1)), axis = 1)
            new_edges = new_edges + offsets[-1]
            edges = np.concatenate((edges, new_edges), axis = 0)
            offsets.append(offsets[-1] + portion.shape[0])

        # add bifurcations
        for junction in self.resampled_junctions:
            node1 = offsets[junction['portion_1']] + junction['point_1']
            node2 = offsets[junction['portion_2']] + junction['point_2']
            edges = np.concatenate((edges, np.array([[node1, node2]])), axis = 0)

        inlet_index = 0
        outlet_indices = [a - 1 for a in offsets[1:]]

        nodes_type = self.stack(self.nodes_type)
        tangents = self.stack(self.tangents)

        return nodes, edges, inlet_index, outlet_indices, nodes_type, tangents

    def create_heterogeneous_graph(self):
        # Dijkstra's algorithm
        def dijkstra_algorithm(nodes, edges, index):
            # make edges bidirectional for simplicity
            nnodes = nodes.shape[0]
            tovisit = np.arange(0,nnodes)
            dists = np.ones((nnodes)) * np.infty
            prevs = np.ones((nnodes)) * (-1)
            b_edges = np.concatenate((edges, \
                      np.array([edges[:,1],edges[:,0]]).transpose()), axis = 0)

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
            return dists, prevs

        nodes, edges, inlet_index, \
        outlet_indices, nodes_type, tangents = self.create_homogeneous_graph()

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
                inlet_physical_contiguous[np.where(inner_mask == \
                                                   edges[iedg,1])[0]] = 1
                break
            if edges[iedg,1] == inlet_index:
                inlet_physical_contiguous[np.where(inner_mask == \
                                                   edges[iedg,0])[0]] = 1
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
            curdistances, _ = dijkstra_algorithm(nodes, edges,\
                                                 outlet_indices[out_index])
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
            outlet_physical_contiguous = np.concatenate((outlet_physical_contiguous,
                                                         cur_opc))

        # we select the edges and properties such that each inner node is
        # only connected to one outlet (based on smaller distance)
        single_connection_mask = []
        for inod in range(nninner_nodes):
            mindist = np.amin(distances_outlets[inod::nninner_nodes])
            indx = np.where(np.abs(distances_outlets - mindist) < 1e-14)[0]
            single_connection_mask.append(int(indx))

        outlet_dict = {'edges': outlet_edges[single_connection_mask,:].astype(int), \
                       'distance': distances_outlets[single_connection_mask], \
                       'x': nodes[outlet_indices,:], \
                       'mask': outlet_mask, \
                       'physical_contiguous': \
                        outlet_physical_contiguous[single_connection_mask].astype(int)}

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
        inner_node_types = np.delete(nodes_type, [inlet_index] + outlet_indices, axis = 0)
        inner_tangents = np.delete(tangents, [inlet_index] + outlet_indices, axis = 0)
        nedges = edges.shape[0]
        inner_pos = np.zeros((nedges, 4))
        for iedg in range(nedges):
            inner_pos[iedg,0:3] = inner_nodes[edges[iedg,1],:] - inner_nodes[edges[iedg,0],:]
            inner_pos[iedg,3] = np.linalg.norm(inner_pos[iedg,0:2])

        inner_dict = {'edges': edges, 'position': inner_pos, 'x': inner_nodes,
                      'mask': inner_mask, 'node_type': inner_node_types,
                      'tangent': inner_tangents}

        return inner_dict, inlet_dict, outlet_dict

    def show(self):
        fig = plt.figure()
        ax = ax = plt.axes(projection='3d')

        for i in range(len(self.portions)):
            portion = self.resampled_portions[i]
            nnodes = portion.shape[0]

            colors = np.zeros((nnodes,3))
            for j in range(nnodes):
                if self.nodes_type[i][j] == 1:
                    colors[j,0] = 1
                elif self.nodes_type[i][j] == 2:
                    colors[j,2] = 1
                elif self.nodes_type[i][j] == 3:
                    colors[j,1] = 1
                elif self.nodes_type[i][j] == 4:
                    colors[j,0] = 0.6
                    colors[j,1] = 0.6
                elif self.nodes_type[i][j] == 5:
                    colors[j,0] = 1
                    colors[j,2] = 0.5
                elif self.nodes_type[i][j] == 6:
                    colors[j,1] = 0.4
                    colors[j,2] = 0.7

            ax.scatter(self.resampled_portions[i][:,0],
                       self.resampled_portions[i][:,1],
                       self.resampled_portions[i][:,2], color = colors)

        plt.show()
