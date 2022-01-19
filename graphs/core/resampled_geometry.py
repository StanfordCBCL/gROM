import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt

class ResampledGeometry:
    def __init__(self, geometry, coeff, remove_caps = True, doresample = True):
        self.geometry = geometry
        self.resample(coeff, doresample)
        self.doresample = doresample
        if remove_caps:
            self.removed_nodes = 3
            self.remove_caps()

        if doresample:
            self.construct_interpolation_matrices()

    def assign_area(self, area):
        self.areas = []
        nportions = len(self.p_portions)
        for ipor in range(0, nportions):
            self.areas.append(self.compute_proj_field(ipor, area))

    def resample(self, coeff, doresample):
        portions = self.geometry.portions
        self.p_portions = []
        for portion in portions:
            p_portion = self.geometry.points[portion[0]:portion[1]+1,:]

            if not doresample:
                self.p_portions.append(p_portion)
            else:
                # compute h of the portion
                alength = 0
                for i in range(0, p_portion.shape[0] - 1):
                    alength += np.linalg.norm(p_portion[i+1,:] - p_portion[i,:])

                N = int(np.floor(alength / (coeff * self.geometry.h)))


                tck, u = scipy.interpolate.splprep([p_portion[:,0],
                                                    p_portion[:,1],
                                                    p_portion[:,2]], s=0, k = 3)
                u_fine = np.linspace(0, 1, N)
                x, y, z = interpolate.splev(u_fine, tck)
                p_portion = np.vstack((x,y,z)).transpose()
                self.p_portions.append(p_portion)

                if p_portion.shape[0] < 2:
                    raise ValueError("Too few points in portion, decrease resample frequency.")

    def construct_interpolation_matrices(self):
        p_matrices = []
        i_matrices = []
        portions = self.geometry.portions
        stdevcoeff = 40

        def kernel(nnorm, h):
            # 99% of the gaussian distribution is within 3 stdev from the mean
            return np.exp(-(nnorm / (2 * (h * stdevcoeff)**2)))

        for ipor in range(0,len(portions)):
            p_portion = self.geometry.points[portions[ipor][0]:portions[ipor][1]+1,:]
            N = self.p_portions[ipor].shape[0]
            M = p_portion.shape[0]
            new_matrix = np.zeros((N,M))

            hs = []
            for j in range(0,M):
                h1 = -1
                h2 = -1
                if j != M-1:
                    h1 = np.linalg.norm(p_portion[j+1,:] - p_portion[j,:])
                if j != 0:
                    h2 = np.linalg.norm(p_portion[j,:] - p_portion[j-1,:])
                h = np.max((h1, h2))
                hs.append(h)
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(self.p_portions[ipor][i,:] -
                                       p_portion[j,:])
                    # we consider 4 stdev to be safe
                    # if n < 4 * hs[j] * stdevcoeff:
                    new_matrix[i,j] = kernel(n,hs[j])

            p_matrices.append(new_matrix)

            N = p_portion.shape[0]
            M = N
            new_matrix = np.zeros((N,M))
            for i in range(0,N):
                for j in range(0,M):
                    n = np.linalg.norm(p_portion[i,:] -  p_portion[j,:])
                    # if n < 4 * hs[j] * stdevcoeff:
                    new_matrix[i,j] = kernel(n,hs[j])

            i_matrices.append(new_matrix)

        self.projection_matrices    = p_matrices
        self.interpolation_matrices = i_matrices

    def plot(self, title = "", field = np.zeros((0))):
        fig = plt.figure(10)
        ax = plt.axes(projection='3d')

        indices = self.geometry.inlet + self.geometry.outlets

        # plot inlet
        ax.scatter3D(self.geometry.points[self.geometry.inlet,0],
                     self.geometry.points[self.geometry.inlet,1],
                     self.geometry.points[self.geometry.inlet,2], color = 'blue')

        # plot outlets
        ax.scatter3D(self.geometry.points[self.geometry.outlets,0],
                     self.geometry.points[self.geometry.outlets,1],
                     self.geometry.points[self.geometry.outlets,2], color = 'red')

        for portion in self.geometry.portions:
            ax.plot3D(self.geometry.points[portion[0]:portion[1]+1,0],
                      self.geometry.points[portion[0]:portion[1]+1,1],
                      self.geometry.points[portion[0]:portion[1]+1,2], 'g--')

        if field.size == 0:
            index = 0
            for portion in self.p_portions:
                # ax.plot3D(portion[:,0], portion[:,1], portion[:,2])
                ax.scatter(portion[:,0], portion[:,1], portion[:,2], color = 'black',
                           s = 0.1)
                N = portion.shape[0]
                ax.text(portion[int(N/2),0],
                        portion[int(N/2),1],
                        portion[int(N/2),2],
                        str(index),
                        color='black',
                        fontsize = 7)
                index = index + 1
        else:
            nportions = len(self.p_portions)

            fmin = np.min(field)
            fmax = np.max(field)

            for ipor in range(0, nportions):
                proj_values = self.compute_proj_field(ipor, field)

                portion = self.p_portions[ipor]
                ax.scatter(portion[:,0], portion[:,1], portion[:,2], s = 2,
                           c = proj_values, vmin = fmin, vmax = fmax)

        # plot bifurcations
        ax.scatter3D(self.geometry.points[self.geometry.bifurcations,0],
                     self.geometry.points[self.geometry.bifurcations,1],
                     self.geometry.points[self.geometry.bifurcations,2], color = 'green')
        plt.title(title)

    def compute_proj_field(self, index_portion, field):
        values = field[self.geometry.portions[index_portion][0]:
                       self.geometry.portions[index_portion][1]+1]
        if self.doresample:
            weights = np.linalg.solve(self.interpolation_matrices[index_portion], values)
            proj_values = np.matmul(self.projection_matrices[index_portion], weights)
        else:
            proj_values = values
            if index_portion == self.inlet:
                proj_values = proj_values[self.removed_nodes:]
            if index_portion in self.outlets:
                proj_values = proj_values[:-self.removed_nodes]

        return proj_values

    def compare_field_along_centerlines(self, field):
        nportions = len(self.p_portions)

        for ipor in range(0, nportions):
            # plot original one
            fig = plt.figure(ipor)
            ax = plt.axes()

            x = [0]
            iin = self.geometry.portions[ipor][0]
            iend = self.geometry.portions[ipor][1]
            points = self.geometry.points[iin:iend+1]

            for ip in range(1, points.shape[0]):
                x.append(np.linalg.norm(points[ip,:] - points[ip-1,:]) + x[ip-1])

            ax.plot(np.array(x), field[iin:iend+1], 'k--o')
            if ipor == 0:
                gf = field[iin:iend+1]
            else:
                gf = np.hstack((gf, field[iin:iend+1]))
            if ipor == 0:
                xsf = np.array(x)
            else:
                xsf = np.hstack((xsf, np.array(x) + xsf[-1]))
            x = [0]
            points = self.p_portions[ipor]

            for ip in range(1, points.shape[0]):
                x.append(np.linalg.norm(points[ip,:] - points[ip-1,:]) + x[ip-1])

            r_field = self.compute_proj_field(ipor, field)
            ax.plot(np.array(x), r_field, 'r-o')
            plt.xlabel('arclength [cm]')
            if ipor == 0:
                gp = r_field
            else:
                gp = np.hstack((gp, r_field))
            if ipor == 0:
                xsp = np.array(x)
            else:
                xsp = np.hstack((xsp, np.array(x) + xsp[-1]))
            # plt.ylabel('flowrate [cm^3/s]')
        fig = plt.figure(100)
        ax = plt.axes()
        ax.plot(xsf, gf, 'k--')
        ax.plot(xsp, gp, 'r-')

    def find_points_type(self, geoarea):
        def circle_intersect_circle(center1, normal1, radius1,
                                    center2, normal2, radius2):

            # then the planes are parallel
            if (np.linalg.norm(center1 - center2) > 1e-12 and
                np.abs(np.dot(normal1, normal2) - 1) < 1e-12):
                return False

            # we find an intersection point between the two planes (we fix x to 0)
            mat = np.zeros((3,3))
            mat[0,:] = normal1
            mat[1,:] = normal2
            mat[2,0] = 1

            b = np.zeros((3))
            b[0] = np.dot(normal1,center1)
            b[1] = np.dot(normal2,center2)

            x = np.linalg.solve(mat, b)

            # line vector of the intersection
            line = np.cross(normal1, normal2)
            line = line / np.linalg.norm(line)

            # find closest point to center1 laying on the line
            t = np.dot(center1 - x, line)
            cp = x + line * t

            # then the point belongs to circle1
            if np.linalg.norm(center1-cp) <= radius1:
                # then the point belongs to circle2
                if np.linalg.norm(center2-cp) <= radius2:
                    return True

            # we do the same for circle 2
            t = np.dot(center2 - x, line)
            cp = x + line * t

            # then the point belongs to circle2
            if np.linalg.norm(center2-cp) <= radius2:
                # then the point belongs to circle1
                if np.linalg.norm(center1-cp) <= radius1:
                    return True

            return False

        nportions = len(self.p_portions)

        allnormals = []
        allradii = []
        for ipor in range(nportions):
            points = self.p_portions[ipor]
            npoints = points.shape[0]

            p_area = self.compute_proj_field(ipor, geoarea)

            # compute normals
            normals = np.zeros(points.shape)

            k = np.min((3, points.shape[0]-1))
            tck, u = scipy.interpolate.splprep([points[:,0],
                                                points[:,1],
                                                points[:,2]], s=0, k = k)

            spline_tangents = scipy.interpolate.splev(u, tck, der=1)

            for i in range(npoints):
                curnormal = np.array([spline_tangents[0][i],
                                      spline_tangents[1][i],
                                      spline_tangents[2][i]])
                normals[i,:] = curnormal / np.linalg.norm(curnormal)

            allnormals.append(normals)
            allradii.append(np.sqrt(p_area/np.pi))

        node_types = []
        for ipor in range(nportions):
            node_types.append(np.zeros((self.p_portions[ipor].shape[0])))

        for ipor in range(nportions):
            for icircle in range(allnormals[ipor].shape[0]):
                for jpor in range(nportions):
                    for jcircle in range(allnormals[jpor].shape[0]):
                        if ipor != jpor or icircle != jcircle:
                            inters = circle_intersect_circle(self.p_portions[ipor][icircle,:],
                                                             allnormals[ipor][icircle,:],
                                                             allradii[ipor][icircle],
                                                             self.p_portions[jpor][jcircle,:],
                                                             allnormals[jpor][jcircle,:],
                                                             allradii[jpor][jcircle])
                            if inters:
                                node_types[ipor][icircle] = 1
                                node_types[jpor][jcircle] = 1

        for ipor in range(nportions):
            if node_types[ipor].shape[0] > 2:
                extr1 = node_types[ipor][0]
                extr2 = node_types[ipor][-1]
                # we apply moving average with window = 3 to avoid situations like 1,0,1,1,1,1
                node_types[ipor] = np.around(np.convolve(node_types[ipor],
                                             np.array([1,1,1]),'same') / 3)
                node_types[ipor][0] = extr1
                node_types[ipor][-1] = extr2

        absorbed_portions = np.ones((nportions)) * -1
        # see if we can merge bifurcations (if one portion is entirely in bif)
        connectivity = np.copy(self.geometry.connectivity)
        for ipor in range(nportions):
            if np.min(node_types[ipor]) == 1:
                bif1 = np.where(connectivity[:,ipor] == 1)[0]
                bif2 = np.where(connectivity[:,ipor] == -1)[0]
                if bif1.size == 0 or bif2.size == 0:
                    print('Warning: portion entirely in junction but it is not inlet and outlet')
                connectivity[bif1,:] = connectivity[bif1,:] + connectivity[bif2,:]
                connectivity = np.delete(connectivity, bif2, axis = 0)
                absorbed_portions[ipor] = bif1

        # we want node_type = 1 for standard 3-way junctions
        degrees = np.sum(np.abs(connectivity),axis = 1).astype(int) - 2
        for ipor in range(nportions):
            points = self.p_portions[ipor]

            # find low extremum
            low_extr = 0
            if absorbed_portions[ipor] == -1:
                degree = degrees[np.where(connectivity[:,ipor] == 1)[0]]
            else:
                # this could fail if more than one bif is merged because their numbering
                # changes
                degree = degrees[int(absorbed_portions[ipor])]
            while low_extr < node_types[ipor].shape[0] and \
                  node_types[ipor][low_extr]== 1:
                node_types[ipor][low_extr] = degree
                low_extr = low_extr + 1

            # find high extremum
            high_extr = node_types[ipor].shape[0] - 1
            if absorbed_portions[ipor] == -1:
                degree = degrees[np.where(connectivity[:,ipor] == -1)[0]]
            else:
                degree = degrees[int(absorbed_portions[ipor])]
            while high_extr >= 0 and node_types[ipor][high_extr] == 1:
                node_types[ipor][high_extr] = degree
                high_extr = high_extr - 1

            node_types[ipor][low_extr:high_extr+1] = 0

        self.node_types = node_types
        self.tangents = allnormals

    def generate_nodes(self):
        nodes = np.zeros((0,3))
        edges = np.zeros((0,2))
        node_type = np.zeros((0,1))
        tangent = np.zeros((0,3))

        inlets = []
        outlets = []
        # we look for inlets and outlets
        njuns = self.geometry.connectivity.shape[0]
        for ijun in range(0, njuns):
            inlets.append(np.where(self.geometry.connectivity[ijun,:] == -1)[0][0])
            outlets.append(np.where(self.geometry.connectivity[ijun,:] == 1)[0])

        outletp = outlets

        offsets = [0]
        nnpoints = []
        isoutlets = []
        for ipor in range(0, len(self.p_portions)):
            isOutlet = False
            for outlet in outlets:
                if ipor in outlet:
                    isOutlet = True
            isoutlets.append(isOutlet)
            if isOutlet:
                nodes = np.vstack((nodes, self.p_portions[ipor][1:,:]))
                node_type = np.vstack((node_type, np.expand_dims(self.node_types[ipor][1:],axis=1)))
                tangent = np.vstack((tangent, self.tangents[ipor][1:,:]))
                npoints = self.p_portions[ipor].shape[0] - 1
            else:
                nodes = np.vstack((nodes, self.p_portions[ipor]))
                node_type = np.vstack((node_type, np.expand_dims(self.node_types[ipor],axis=1)))
                tangent = np.vstack((tangent, self.tangents[ipor]))
                npoints = self.p_portions[ipor].shape[0]

            for i in range(0, npoints - 1):
                edges = np.vstack((edges, np.array((i + offsets[ipor],
                                                    i + offsets[ipor] + 1))))

            nnpoints.append(npoints)
            offsets.append(offsets[ipor] + npoints)

        # add junctions to the edges
        for ijun in range(0, njuns):
            inlet = np.where(self.geometry.connectivity[ijun,:] == -1)[0][0]
            outlets = np.where(self.geometry.connectivity[ijun,:] == 1)[0]
            for ioutlet in range(0, outlets.size):
                outlet = outlets[ioutlet]
                edges = np.vstack((edges, np.array((offsets[inlet] + nnpoints[inlet] - 1,
                                                   offsets[outlet]))))

        lengths = []
        for edge in edges:
            lengths.append(np.linalg.norm(nodes[int(edge[1]),:]-nodes[int(edge[0]),:]))

        self.offsets = offsets
        self.npoints = nnpoints
        self.isoutlets = isoutlets

        nportions = len(self.p_portions)

        inlet_node = 0
        outlet_nodes = []
        for i in range(0,nportions):
            if i not in inlets:
                outlet_nodes.append(self.offsets[i] + self.npoints[i] - 1)

            isinlet = True
            for outlet in outletp:
                if i in outlet:
                    isinlet = False

            if isinlet:
                inlet_node = self.offsets[i]

        edges = edges.astype(np.int)
        return nodes, edges, lengths, inlet_node, outlet_nodes, node_type, tangent

    def compute_graph_field(self, field):
        if not hasattr(self, 'isoutlets'):
            self.generate_nodes()

        newfield = np.zeros((0,1))
        selectors = []
        for ipor in range(0, len(self.p_portions)):
            f = self.compute_proj_field(ipor, field)

            if self.isoutlets[ipor]:
                newfield = np.vstack((newfield, np.expand_dims(f[1:], axis = 1)))
            else:
                newfield = np.vstack((newfield, np.expand_dims(f, axis = 1)))

        return newfield.astype(np.float64)

    def generate_fields(self, pressures, velocities, areas):

        g_pressures = {}
        g_velocities = {}


        for t in pressures:
            g_pressures[t] = self.compute_graph_field(pressures[t])
            g_velocities[t] = self.compute_graph_field(velocities[t])

        return g_pressures, g_velocities, self.compute_graph_field(areas)

    def remove_caps(self):
        connectivity = self.geometry.connectivity

        inlet = 0
        outlets = []
        for jpor in range(connectivity.shape[1]):
            if np.sum(connectivity[:,jpor]) == -1:
                inlet = jpor
            if np.sum(connectivity[:,jpor]) == 1:
                outlets.append(jpor)

        if len(outlets) == 0:
            outlets.append(0)

        self.inlet = inlet
        self.outlets = outlets

        for ipor in range(len(self.p_portions)):
            if ipor == inlet:
                self.p_portions[ipor] = self.p_portions[ipor][self.removed_nodes:,:]
            if ipor in outlets:
                self.p_portions[ipor] = self.p_portions[ipor][:-self.removed_nodes,:]
