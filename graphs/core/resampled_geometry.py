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

        if remove_caps:
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

    def generate_nodes(self):
        nodes = np.zeros((0,3))
        edges = np.zeros((0,2))

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
                npoints = self.p_portions[ipor].shape[0] - 1
            else:
                nodes = np.vstack((nodes, self.p_portions[ipor]))
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

        return nodes.astype(np.float64), edges, lengths, inlet_node, outlet_nodes

    def generate_fields(self, pressures, velocities, areas):
        if not hasattr(self, 'isoutlets'):
            self.generate_nodes()

        g_pressures = {}
        g_velocities = {}

        def compute_g_field(field):
            newfield = np.zeros((0,1))

            for ipor in range(0, len(self.p_portions)):
                f = self.compute_proj_field(ipor, field)

                from scipy.signal import savgol_filter
                # f = savgol_filter(f, int(np.min((11, 2*np.floor(f.size/2)-1))), 3)

                if self.isoutlets[ipor]:
                    newfield = np.vstack((newfield, np.expand_dims(f[1:], axis = 1)))
                else:
                    newfield = np.vstack((newfield, np.expand_dims(f, axis = 1)))
                newfield
            return newfield.astype(np.float64)

        for t in pressures:
            g_pressures[t] = compute_g_field(pressures[t])
            g_velocities[t] = compute_g_field(velocities[t])

        return g_pressures, g_velocities, compute_g_field(areas)

    def remove_caps(self):
        connectivity = self.geometry.connectivity

        inlet = 0
        outlets = []
        for jpor in range(connectivity.shape[0]):
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
