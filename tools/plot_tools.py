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

def circle_intersect_circle(circle1, circle2):
    center1 = np.mean(circle1, axis=0)

    normal1 = np.cross((circle1[0,:]-center1),(circle1[1,:]-center1))
    normal1 = normal1 / np.linalg.norm(normal1)

    center2 = np.mean(circle2, axis=0)

    normal2 = np.cross((circle2[0,:]-center2),(circle2[1,:]-center2))
    normal2 = normal2 / np.linalg.norm(normal2)

    radius1 = np.linalg.norm((circle1[0,:]-center1))
    radius2 = np.linalg.norm((circle2[0,:]-center2))

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

    # then the point belongs to circle1
    if np.linalg.norm(center2-cp) <= radius2:
        # then the point belongs to circle2
        if np.linalg.norm(center1-cp) <= radius1:
            return True

    return False

def test_circle_intersect_circle():
    theta = np.linspace(0, 2 * np.pi, 100)
    theta = theta[:-1]

    circle1 = np.array([np.sin(theta),np.cos(theta),theta * 0]).transpose()
    circle2 = np.array([0 * theta,np.sin(theta),np.cos(theta)]).transpose()

    circle_intersect_circle(circle1, circle2)

def plot3Dgeo(geometry, area, downsampling_ratio):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax._axis3don = False

    nportions = len(geometry.p_portions)

    minarea = np.min(area)
    maxarea = np.max(area)

    minx = np.infty
    maxx = np.NINF
    miny = np.infty
    maxy = np.NINF
    minz = np.infty
    maxz = np.NINF

    allcircles = []
    allareas = []
    for ipor in range(nportions):
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
            # plt.plot(circle[:,0], circle[:,1], circle[:,2], c = 'blue')
            circles.append(circle)
            areas.append(area[i])

        allcircles.append(circles)
        allareas.append(areas)

        minx = np.min([minx,np.min(points[:,0])])
        maxx = np.max([maxx,np.max(points[:,0])])
        miny = np.min([miny,np.min(points[:,1])])
        maxy = np.max([maxy,np.max(points[:,1])])
        minz = np.min([minz,np.min(points[:,2])])
        maxz = np.max([maxz,np.max(points[:,2])])

    # test_circle_intersect_circle()

    delete_circles = [set() for i in range(nportions)]
    for ipor in range(nportions):
        icircles = allcircles[ipor]
        for icircle in range(len(icircles)):
            for jpor in range(nportions):
                jcircles = allcircles[jpor]
                for jcircle in range(len(jcircles)):
                    if ipor != jpor or icircle != jcircle:
                        inters = circle_intersect_circle(icircles[icircle],
                                                         jcircles[jcircle])
                        if inters:
                            delete_circles[ipor].add(icircle)
                            delete_circles[jpor].add(jcircle)

    for ipor in range(nportions):
        curset = delete_circles[ipor]
        offset = 0
        for icircle in curset:
            del allcircles[ipor][icircle-offset]
            offset = offset+1

    for ipor in range(nportions):
        circles = allcircles[ipor]
        if len(circles) > 0:
            areas = allareas[ipor]
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

geo = ResampledGeometry(Geometry(p_array), 5, remove_caps = True, doresample = True)

plot3Dgeo(geo, fields['area'], downsampling_ratio = 5)
