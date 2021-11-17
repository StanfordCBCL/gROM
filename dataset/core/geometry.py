import numpy as np
import matplotlib.pyplot as plt

class Geometry:
    def __init__(self, points):
        self.points = points
        self.inlet  = [0]
        self.find_characteristic_h()
        self.find_outlets()
        self.find_bifurcations()
        self.find_portions()
        # adjust mesh size
        self.find_characteristic_h()
        # merge bifurcations close to each other
        self.simplify()
        self.find_connectivity()

    def find_characteristic_h(self):
        if not hasattr(self, 'portions'):
            # first approximation of characteristic h
            self.h = np.linalg.norm(self.points[1,:] - self.points[0,:])
        else:
            hsum = 0
            count = 0
            for portion in self.portions:
                for i in range(portion[0],portion[1]):
                    hsum += np.linalg.norm(self.points[i+1] - self.points[i])
                    count = count + 1
            self.h = hsum / count
        # tolerance to determine if two points are neighbors
        self.tol = 3 * self.h

    def find_outlets(self):
        npoints = self.points.shape[0]
        outlets = []
        for i in range(0, npoints-1):
            curh = np.linalg.norm(self.points[i+1,:] - self.points[i,:])
            if curh > self.tol:
                outlets.append(i)

        # we assume that the last point is an outlet and the first is an inlet
        outlets.append(npoints-1)
        self.outlets = outlets

    def find_closest_previous_point(self, ipoint):
        chunk = self.points[:ipoint,:]
        diff = np.subtract(chunk, self.points[ipoint,:])
        nn = np.linalg.norm(diff, axis = 1)
        return np.where(nn == nn.min())[0][0]

    def find_bifurcations(self):
        bifurcations = []
        for outlet in self.outlets[:-1]:
          bifurcations.append(self.find_closest_previous_point(outlet + 1))

        self.bifurcations = bifurcations

    def find_portions(self):
        portions = []
        indices = self.inlet + self.outlets + self.bifurcations
        indices.sort()

        for i in range(0,len(indices)-1):
            if indices[i] in self.outlets:
                # portions.append([indices[i]+1, indices[i+1]])
                portions.append([indices[i]+1, indices[i+1]])
            else:
                portions.append([indices[i], indices[i+1]])

        self.portions = portions

    def simplify(self):
        self.tol_simplify = self.tol * 5
        portions_c = self.portions.copy()
        for portion in portions_c:
            n = np.linalg.norm(self.points[portion[1],:] - self.points[portion[0],:])
            # this might be problematic
            if n < self.tol_simplify:
                self.portions.remove(portion)

        bifurcations_c = self.bifurcations.copy()
        for ibif in range(0, len(bifurcations_c)):
            bi = bifurcations_c[ibif]
            for jbif in range(ibif+1, len(bifurcations_c)):
                bj = bifurcations_c[jbif]
                n = np.linalg.norm(self.points[bi,:] - self.points[bj,:])
                if n < self.tol_simplify:
                    self.bifurcations.remove(bj)

    def find_connectivity(self):
        nbif = len(self.bifurcations)
        npor = len(self.portions)

        connectivity = np.zeros((nbif, npor))

        for i in range(0, nbif):
            for j in range(0, npor):
                if np.linalg.norm(self.points[self.bifurcations[i],:] -
                                  self.points[self.portions[j][0],:]) < self.tol_simplify:
                    connectivity[i,j] = 1
                elif np.linalg.norm(self.points[self.bifurcations[i],:] -
                                    self.points[self.portions[j][1],:]) < self.tol_simplify:
                    connectivity[i,j] = -1

        self.p_inlet   = []
        self.p_outlets = []

        for iportion in range(0, len(self.portions)):
            if self.inlet[0] in self.portions[iportion]:
                self.p_inlet.append(iportion)

            for outlet in self.outlets:
                if outlet in self.portions[iportion]:
                    self.p_outlets.append(iportion)

        self.connectivity = connectivity

    def plot(self, title = "", field = np.zeros((0))):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        indices = self.inlet + self.outlets

        # plot inlet
        ax.scatter3D(self.points[self.inlet,0],
                     self.points[self.inlet,1],
                     self.points[self.inlet,2], color = 'blue')

        # plot outlets
        ax.scatter3D(self.points[self.outlets,0],
                     self.points[self.outlets,1],
                     self.points[self.outlets,2], color = 'red')

        if field.size == 0:
            for portion in self.portions:
                ax.plot3D(self.points[portion[0]:portion[1]+1,0],
                          self.points[portion[0]:portion[1]+1,1],
                          self.points[portion[0]:portion[1]+1,2])
        else:
            fmin = np.min(field)
            fmax = np.max(field)

            for portion in self.portions:
                values = field[portion[0]:portion[1]+1]

                ax.scatter(self.points[portion[0]:portion[1]+1,0],
                           self.points[portion[0]:portion[1]+1,1],
                           self.points[portion[0]:portion[1]+1,2], s = 2,
                           c = values, vmin = fmin, vmax = fmax)
        # plot bifurcations
        ax.scatter3D(self.points[self.bifurcations,0],
                     self.points[self.bifurcations,1],
                     self.points[self.bifurcations,2], color = 'green')
        plt.title(title)
