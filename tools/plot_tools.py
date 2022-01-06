import sys

sys.path.append("../graphs/core")

import io_utils as io
from geometry import Geometry
from resampled_geometry import ResampledGeometry
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import interpolate
import geomdl.fitting
from geomdl.visualization import VisMPL
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def circle3D(center, normal, radius, npoints):
    theta = np.linspace(0, 2 * np.pi, npoints)

    axis = np.array([normal[2],0,-normal[0]])
    axis = axis / np.linalg.norm(axis)
    if (axis[0] < 0):
        axis = - axis

    axis_p = np.cross(normal, axis)
    axis_p = axis_p / np.linalg.norm(axis_p)
    if (axis_p[1] < 0):
        axis_p = - axis_p

    circle = np.zeros((npoints, 3))

    for ipoint in range(npoints):
        circle[ipoint,:] = center + np.sin(theta[ipoint]) * axis * radius + np.cos(theta[ipoint]) * axis_p * radius

    return circle



    # circle2D = np.zeros((npoints,3))
    # circle2D[:,0] = np.sin(points) * radius
    # circle2D[:,1] = np.cos(points) * radius
    # circle2D[:,2] = np.zeros((npoints))
    #
    # normal2d = np.array([0, 0, 1])
    #
    # # compute rotation matrix like this https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    # v = np.cross(normal, normal2d)
    #
    # V = np.zeros((3,3))
    # V[0,1] = -v[2]
    # V[0,2] = v[1]
    # V[1,0] = v[2]
    # V[1,2] = -v[0]
    # V[2,0] = -v[1]
    # V[2,1] = v[0]
    #
    # c = np.dot(normal, normal2d)
    # R = np.eye(3) + V + np.matmul(V, V) * (1 / (1 + c))
    #
    # # we premultiply R such that (1,0,0) in 2D remains with 0-y component (to
    # # avoid incompatible parametrization between neighbors)
    #
    # theta = np.arctan(-R[1,1]/R[1,0])
    # A = np.sin(theta)
    # B = np.cos(theta)
    #
    # R2D = np.zeros((3,3))
    # R2D[2,2] = 1
    # R2D[0,0] = A
    # R2D[1,0] = -B
    # R2D[0,1] = B
    # R2D[1,1] = A
    #
    # R1 = np.matmul(R,R2D)
    # if R1[0,0] < 0:
    #     theta = theta + np.pi
    #     A = np.sin(theta)
    #     B = np.cos(theta)
    #     R2D[0,0] = A
    #     R2D[1,0] = -B
    #     R2D[0,1] = B
    #     R2D[1,1] = A
    #     R = np.matmul(R,R2D)
    # else:
    #     R = R1
    #
    # return np.matmul(circle2D, R.transpose()) + center


def plot3Dgeo(geometry, area, downsampling_ratio):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    nportions = len(geometry.p_portions)

    minarea = np.min(area)
    maxarea = np.max(area)

    minx = np.infty
    maxx = np.NINF
    miny = np.infty
    maxy = np.NINF
    minz = np.infty
    maxz = np.NINF

    for ipor in range(nportions):
    # for ipor in range(6,7):
        points = geometry.p_portions[ipor]
        npoints = points.shape[0]

        area = geo.compute_proj_field(ipor, fields['area'])

        # compute normals
        normals = np.zeros(points.shape)

        tck, u = scipy.interpolate.splprep([points[:,0],
                                            points[:,1],
                                            points[:,2]], s=0, k = 3)

        spline_normals = scipy.interpolate.splev(u, tck, der=1)

        for i in range(npoints):
            curnormal = np.array([spline_normals[0][i],spline_normals[1][i],spline_normals[2][i]])
            normals[i,:] = curnormal / np.linalg.norm(curnormal)

        plt.plot(points[:,0], points[:,1], points[:,2])

        n = np.max((2, int(npoints/downsampling_ratio)))
        actualidxs = np.floor(np.linspace(0,npoints-1,n)).astype(int)
        circles = []
        areas = []
        ncircle_points = 60
        for i in actualidxs:
            circle = circle3D(points[i,:], normals[i,:], np.sqrt(area[i]/np.pi), ncircle_points)
            plt.plot(circle[:,0], circle[:,1], circle[:,2], c = 'blue')
            circles.append(circle)
            areas.append(area[i])

        ncircles = len(circles)
        nsubs = 20
        X = np.zeros((ncircle_points,(ncircles - 1) * nsubs + 1))
        Y = np.zeros((ncircle_points,(ncircles - 1) * nsubs + 1))
        Z = np.zeros((ncircle_points,(ncircles - 1) * nsubs + 1))
        C = np.zeros((ncircle_points,(ncircles - 1) * nsubs + 1))
        coefs = np.linspace(0,1,nsubs+1)
        for i in range(ncircles-1):
            for j in range(nsubs):
                X[:,i * nsubs + j] = circles[i][:,0] * (1 - coefs[j]) + circles[i+1][:,0] * coefs[j]
                Y[:,i * nsubs + j] = circles[i][:,1] * (1 - coefs[j]) + circles[i+1][:,1] * coefs[j]
                Z[:,i * nsubs + j] = circles[i][:,2] * (1 - coefs[j]) + circles[i+1][:,2] * coefs[j]
                C[:,i * nsubs + j] = areas[i] * (1 - coefs[j]) + areas[i+1] * coefs[j]

        X[:,-1] = circles[-1][:,0]
        Y[:,-1] = circles[-1][:,1]
        Z[:,-1] = circles[-1][:,2]
        C[:,-1] = areas[-1]

        colors = (C - minarea) / (maxarea - minarea)

        cmap = cm.get_cmap("coolwarm")
        ax.plot_surface(X,Y,Z, facecolors=cmap(colors), shade=True, alpha=0.8)

        minx = np.min([minx,np.min(points[:,0])])
        maxx = np.max([maxx,np.max(points[:,0])])
        miny = np.min([miny,np.min(points[:,1])])
        maxy = np.max([maxy,np.max(points[:,1])])
        minz = np.min([minz,np.min(points[:,2])])
        maxz = np.max([maxz,np.max(points[:,2])])

    m = np.min([minx,miny,minz])
    M = np.max([maxx,maxy,maxz])

    padding = np.max([np.abs(m),np.abs(M)]) * 0.1

    minx = minx - padding
    maxx = maxx + padding
    miny = miny - padding
    maxy = maxy + padding
    minz = minz - padding
    maxz = maxz + padding

    ax.set_box_aspect((maxx-minx, maxy-miny, maxz-minz))

    ax.set_xlim((minx,maxx))
    ax.set_ylim((miny,maxy))
    ax.set_zlim((minz,maxz))

    plt.show()

model_name = '0063_1001'
print('Create geometry: ' + model_name)
soln = io.read_geo('../graphs/vtps/' + model_name + '.vtp').GetOutput()
fields, _, p_array = io.get_all_arrays(soln)

geo = ResampledGeometry(Geometry(p_array), 5, True, True)

plot3Dgeo(geo, fields['area'], downsampling_ratio = 5)
