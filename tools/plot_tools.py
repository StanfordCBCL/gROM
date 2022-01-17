import sys

sys.path.append("../graphs/core")
sys.path.append("../network/")

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
from matplotlib import animation
import preprocessing as pp

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

def plot_3D_graph(graph, state = None, coefs = None, field_name = None, cmap = cm.get_cmap("viridis")):
    fig = plt.figure()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.axes(projection='3d')
    ax._axis3don = False

    scatterpts = None
    if state == None:
        # plot inlet
        xin = graph.nodes['inlet'].data['x'].detach().numpy()
        ax.scatter(xin[:,0], xin[:,1], xin[:,2], color='red', depthshade=0)

        # plot outlet
        xout = graph.nodes['outlet'].data['x'].detach().numpy()
        ax.scatter(xout[:,0], xout[:,1], xout[:,2], color='blue', depthshade=0)

        # plot inner
        xinner = graph.nodes['inner'].data['x'].detach().numpy()
        ax.scatter(xinner[:,0], xinner[:,1], xinner[:,2], color='black', depthshade=0)

        x = np.concatenate((xin,xout,xinner),axis=0)

    else:
        xin = graph.nodes['inlet'].data['x'].detach().numpy()
        xout = graph.nodes['outlet'].data['x'].detach().numpy()
        xinner = graph.nodes['inner'].data['x'].detach().numpy()
        x = np.concatenate((xin,xout,xinner),axis=0)

        fin = state[field_name]['inlet']
        fout = state[field_name]['outlet']
        finner = state[field_name]['inner']
        field = np.concatenate((fin,fout,finner),axis=0)

        field = pp.invert_normalize_function(field, field_name, coefs)

        C = (field - coefs[field_name]['min']) / \
            (coefs[field_name]['max'] - coefs[field_name]['min'])

        scatterpts = ax.scatter(x[:,0], x[:,1], x[:,2],
                                color=cmap(C), depthshade=0)

    minx = np.min(x, axis=0)
    maxx = np.max(x, axis=0)

    m = np.min(minx)
    M = np.max(maxx)

    padding = np.max([np.abs(m),np.abs(M)]) * 0.1

    minx = minx - padding
    maxx = maxx + padding

    ax.set_box_aspect((maxx[0]-minx[0], maxx[1]-minx[1], maxx[2]-minx[2]))

    ax.set_xlim((minx[0],maxx[0]))
    ax.set_ylim((minx[1],maxx[1]))
    ax.set_zlim((minx[2],maxx[2]))

    cbar = fig.colorbar(scatterpts)

    cbar.set_ticks([0, 1])
    if field_name == 'pressure':
        minv = str('{:.0f}'.format(float(coefs[field_name]['min']/1333.2)))
        maxv = str('{:.0f}'.format(float(coefs[field_name]['max']/1333.2)))
    else:
        minv = str('{:.0f}'.format(float(coefs[field_name]['min'])))
        maxv = str('{:.0f}'.format(float(coefs[field_name]['max'])))

    cbar.set_ticklabels([minv, maxv])

    return fig, ax, scatterpts

def plot3Dgeo(geometry, area, downsampling_ratio, field, minvalue, maxvalue,
              nsubs = 20, cmap = cm.get_cmap("coolwarm")):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax._axis3don = False

    geoarea = area

    nportions = len(geometry.p_portions)

    minarea = np.min(area)
    maxarea = np.max(area)

    minx = np.infty
    maxx = np.NINF
    miny = np.infty
    maxy = np.NINF
    minz = np.infty
    maxz = np.NINF

    allactualidxs = []
    allcircles = []
    allareas = []
    for ipor in range(nportions):
        points = geometry.p_portions[ipor]
        npoints = points.shape[0]

        area = geometry.compute_proj_field(ipor, geoarea)

        # compute normals
        normals = np.zeros(points.shape)

        tck, u = scipy.interpolate.splprep([points[:,0],
                                            points[:,1],
                                            points[:,2]], s=0, k = 3)

        spline_normals = scipy.interpolate.splev(u, tck, der=1)

        for i in range(npoints):
            curnormal = np.array([spline_normals[0][i],spline_normals[1][i],spline_normals[2][i]])
            normals[i,:] = curnormal / np.linalg.norm(curnormal)

        n = np.max((2, int(npoints/downsampling_ratio)))
        actualidxs = np.floor(np.linspace(0,npoints-1,n)).astype(int)
        allactualidxs.append(actualidxs)
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

    node_types = []
    for ipor in range(nportions):
        node_types.append(np.zeros((geometry.p_portions[ipor].shape[0])))

    delete_circles = [set() for i in range(nportions)]
    redpoints = np.zeros((0,3))
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
                            node_types[ipor][icircle] = 1
                            node_types[jpor][jcircle] = 1

    absorbed_portions = np.ones((nportions)) * -1
    # see if we can merge bifurcations (if one portion is entirely in bif)
    connectivity = np.copy(geometry.geometry.connectivity)
    for ipor in range(nportions):
        if np.min(node_types[ipor]) == 1:
            bif1 = np.where(connectivity[:,ipor] == 1)[0]
            bif2 = np.where(connectivity[:,ipor] == -1)[0]
            if bif1.size == 0 or bif2.size == 0:
                print('Warning: portion entirely in junction but it is not inlet and outlet')
            connectivity[bif1,:] = connectivity[bif1,:] + connectivity[bif2,:]
            connectivity = np.delete(connectivity, bif2, axis = 0)
            absorbed_portions[ipor] = bif1

    degrees = np.sum(np.abs(connectivity),axis = 1).astype(int)
    print(absorbed_portions)
    for ipor in range(nportions):
        points = geometry.p_portions[ipor]

        # find low extremum
        low_extr = 0
        if absorbed_portions[ipor] == -1:
            degree = degrees[np.where(connectivity[:,ipor] == 1)[0]]
            print(connectivity)
        else:
            # this could fail if more than one bif is merged because their numbering
            # changes
            print(absorbed_portions[ipor])
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

        colors = np.zeros((geometry.p_portions[ipor].shape[0],3))

        for i in range(node_types[ipor].shape[0]):
            if node_types[ipor][i] == 3:
                colors[i] = np.array([1,0,0])
            if node_types[ipor][i] == 5:
                colors[i] = np.array([0,1,0])

        ax.scatter(points[:,0], points[:,1], points[:,2], color = colors, s = 2)

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

def plot_linear(pressures_pred, flowrates_pred, pressures_real, flowrates_real, times,
                coefs_dict, outfile_name, time, framerate = 60):

    nframes = time * framerate
    indices = np.floor(np.linspace(0,len(pressures_pred)-1,nframes)).astype(int)

    selected_times = []
    selected_pred_pressure = []
    selected_real_pressure = []
    selected_pred_flowrate = []
    selected_real_flowrate = []
    for ind in indices:
        selected_pred_pressure.append(pressures_pred[ind])
        selected_real_pressure.append(pressures_real[ind])
        selected_pred_flowrate.append(flowrates_pred[ind])
        selected_real_flowrate.append(flowrates_real[ind])
        selected_times.append(times[0,ind])

    pressures_pred = selected_pred_pressure
    pressures_real = selected_real_pressure
    flowrates_pred = selected_pred_flowrate
    flowrates_real = selected_real_flowrate
    times = selected_times

    fig, ax = plt.subplots(2)
    line_pred_p, = ax[0].plot([],[],'r')
    line_pred_p.set_label('GNN')
    line_real_p, = ax[0].plot([],[],'--b')
    line_real_p.set_label('Ground truth')
    line_pred_q, = ax[1].plot([],[],'r')
    line_pred_q.set_label('GNN')
    line_real_q, = ax[1].plot([],[],'--b')
    line_real_q.set_label('Ground truth')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')

    ax[1].set_xlabel('graph node index')
    ax[0].set_ylabel('pressure [mmHg]')
    ax[1].set_ylabel('flowrate [cc^3/s]')

    def animation_frame(i):
        line_pred_p.set_xdata(range(0,len(pressures_pred[i])))
        line_pred_p.set_ydata(pp.invert_normalize_function(pressures_pred[i],'pressure',coefs_dict) / 1333.2)
        line_real_p.set_xdata(range(0,len(pressures_pred[i])))
        line_real_p.set_ydata(pp.invert_normalize_function(pressures_real[i],'pressure',coefs_dict) / 1333.2)
        line_pred_q.set_xdata(range(0,len(flowrates_pred[i])))
        line_pred_q.set_ydata(pp.invert_normalize_function(flowrates_pred[i],'flowrate',coefs_dict))
        line_real_q.set_xdata(range(0,len(flowrates_pred[i])))
        line_real_q.set_ydata(pp.invert_normalize_function(flowrates_real[i],'flowrate',coefs_dict))
        ax[0].set_title('{:.2f} s'.format(float(times[i])))
        ax[0].set_xlim(0,len(pressures_pred[i]))
        ax[0].set_ylim((coefs_dict['pressure']['min']-np.abs(coefs_dict['pressure']['min'])*0.1)/1333.2,(coefs_dict['pressure']['max']+np.abs(coefs_dict['pressure']['max'])*0.1) / 1333.2)
        ax[1].set_xlim(0,len(flowrates_pred[i]))
        ax[1].set_ylim(coefs_dict['flowrate']['min']-np.abs(coefs_dict['flowrate']['min'])*0.1,coefs_dict['flowrate']['max']+np.abs(coefs_dict['flowrate']['max'])*0.1)
        return line_pred_p, line_real_p, line_pred_q, line_real_q

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(pressures_pred),
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=framerate)
    anim.save(outfile_name, writer = writervideo)

def plot_linear_inlet(graph, pred_states, real_states, coefs_dict, field_name):

    fig, ax = plt.subplots(1)

    physical_contiguous = graph.edges['in_to_inner'].data['physical_contiguous']
    indnext = np.where(physical_contiguous == 1)[0]

    edges = graph.edges['inner_to_inner'].data['edges'].detach().numpy()
    ntype = graph.nodes['inner'].data['node_type'].detach().numpy()
    nodes = graph.nodes['inner'].data['x'].detach().numpy()

    node_selector = [indnext[0]]
    dists = [0, np.linalg.norm(nodes[indnext,:] - graph.nodes['inlet'].data['x'].detach().numpy())]
    stop = False
    while not stop:
        next = edges[np.where(edges[:,0] == node_selector[-1])[0],1].tolist()
        for ns in node_selector:
            if ns in next:
                next.remove(ns)
        if len(next) > 1:
            stop = True
            break
        node_selector.append(next[0])
        dists.append(dists[-1] + np.linalg.norm(nodes[next,:] - nodes[node_selector[-1],:]))

    pred_field = np.concatenate((np.reshape(pred_states[field_name]['inlet'].detach().numpy(),(1)),
                                 pred_states[field_name]['inner'].detach().numpy()[node_selector,0]),axis=0)
    real_field = np.concatenate((np.reshape(real_states[field_name]['inlet'].detach().numpy(),(1)),
                                 real_states[field_name]['inner'].detach().numpy()[node_selector,0]),axis=0)

    pred_field = pp.invert_normalize_function(pred_field, field_name, coefs_dict)
    real_field = pp.invert_normalize_function(real_field, field_name, coefs_dict)

    line_pred, = ax.plot(dists,pred_field,'r')
    line_real, = ax.plot(dists,real_field,'--b')

    ax.set_xlim(0,dists[-1])
    ax.set_ylim(coefs_dict[field_name]['min'],coefs_dict[field_name]['max'])

    return fig, ax, line_pred, line_real, node_selector

def plot_linear_outlet(graph, pred_states, real_states, coefs_dict, field_name, out_index):

    fig, ax = plt.subplots(1)

    outedges = graph.edges['out_to_inner'].data['edges']
    outedgesind = np.where(outedges[:,0] == out_index)[0]

    B = np.expand_dims(graph.edges['out_to_inner'].data['physical_contiguous'],axis=1)
    aggr = np.concatenate((graph.edges['out_to_inner'].data['edges'],B),axis=1)

    physical_contiguous = aggr[outedgesind]
    indnext = physical_contiguous[np.where(physical_contiguous[:,2] == 1)[0],1]

    edges = graph.edges['inner_to_inner'].data['edges'].detach().numpy()
    ntype = graph.nodes['inner'].data['node_type'].detach().numpy()
    nodes = graph.nodes['inner'].data['x'].detach().numpy()

    node_selector = [indnext[0]]
    stop = False
    while not stop:
        next = edges[np.where(edges[:,0] == node_selector[-1])[0],1].tolist()
        for ns in node_selector:
            if ns in next:
                next.remove(ns)
        if len(next) > 1:
            stop = True
            break
        if len(next) == 0:
            print('hello')
        node_selector.append(next[0])

    node_selector.reverse()

    dists = np.zeros((len(node_selector) + 1))

    for i in range(1,len(node_selector)):
        dists[i] = dists[i-1] + np.linalg.norm(nodes[node_selector[i],:]-nodes[node_selector[i-1],:])

    dists[-1] = dists[-2] + np.linalg.norm(nodes[node_selector[-1],:]-graph.nodes['outlet'].data['x'][out_index].detach().numpy())

    pred_field = np.concatenate((pred_states[field_name]['inner'].detach().numpy()[node_selector,0],
                                 np.reshape(pred_states[field_name]['outlet'][out_index].detach().numpy(),(1))),axis=0)
    real_field = np.concatenate((real_states[field_name]['inner'].detach().numpy()[node_selector,0],
                                 np.reshape(pred_states[field_name]['outlet'][out_index].detach().numpy(),(1))),axis=0)

    pred_field = pp.invert_normalize_function(pred_field, field_name, coefs_dict)
    real_field = pp.invert_normalize_function(real_field, field_name, coefs_dict)

    line_pred, = ax.plot(dists,pred_field,'r')
    line_real, = ax.plot(dists,real_field,'--b')

    ax.set_xlim(0,dists[-1])
    ax.set_ylim(coefs_dict[field_name]['min'],coefs_dict[field_name]['max'])

    return fig, ax, line_pred, line_real, node_selector

def plot_inlet(model_name, pred_states, real_states, times,
               coefs_dict, field_name, outfile_name, time, framerate = 60):

    nframes = time * framerate
    indices = np.floor(np.linspace(0,len(pred_states)-1,nframes)).astype(int)

    selected_times = []
    selected_pred_states = []
    selected_real_states = []
    for ind in indices:
        selected_pred_states.append(pred_states[ind])
        selected_real_states.append(real_states[ind])
        selected_times.append(times[0,ind])

    pred_states = selected_pred_states
    real_states = selected_real_states
    times = selected_times

    graph = pp.load_graphs('../graphs/data/' + model_name + '.grph')[0][0]

    fig, ax, line_pred, line_real, node_selector = plot_linear_inlet(graph,
                                                pred_states[0], real_states[0],
                                                coefs_dict, field_name)

    def animation_frame(i):
        pred_state = pred_states[i]
        real_state = real_states[i]

        pred_field = np.concatenate((np.reshape(pred_state[field_name]['inlet'].detach().numpy(),(1)),
                                     pred_state[field_name]['inner'].detach().numpy()[node_selector,0]),axis=0)
        real_field = np.concatenate((np.reshape(real_state[field_name]['inlet'].detach().numpy(),(1)),
                                     real_state[field_name]['inner'].detach().numpy()[node_selector,0]),axis=0)

        pred_field = pp.invert_normalize_function(pred_field, field_name, coefs_dict)
        real_field = pp.invert_normalize_function(real_field, field_name, coefs_dict)

        line_pred.set_ydata(pred_field)
        line_real.set_ydata(real_field)

        ax.set_title('{:.2f} s'.format(float(times[i])))
        return

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(pred_states),
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=framerate)
    anim.save(outfile_name, writer = writervideo)

def plot_outlet(model_name, pred_states, real_states, times,
               coefs_dict, field_name, outfile_name, out_index, time, framerate = 60):

    nframes = time * framerate
    indices = np.floor(np.linspace(0,len(pred_states)-1,nframes)).astype(int)

    selected_times = []
    selected_pred_states = []
    selected_real_states = []
    for ind in indices:
        selected_pred_states.append(pred_states[ind])
        selected_real_states.append(real_states[ind])
        selected_times.append(times[0,ind])

    pred_states = selected_pred_states
    real_states = selected_real_states
    times = selected_times

    graph = pp.load_graphs('../graphs/data/' + model_name + '.grph')[0][0]

    fig, ax, line_pred, line_real, node_selector = plot_linear_outlet(graph,
                                                pred_states[0], real_states[0],
                                                coefs_dict, field_name, out_index)

    def animation_frame(i):
        pred_state = pred_states[i]
        real_state = real_states[i]

        pred_field = np.concatenate((np.reshape(pred_state[field_name]['inlet'].detach().numpy(),(1)),
                                     pred_state[field_name]['inner'].detach().numpy()[node_selector,0]),axis=0)
        real_field = np.concatenate((np.reshape(real_state[field_name]['inlet'].detach().numpy(),(1)),
                                     real_state[field_name]['inner'].detach().numpy()[node_selector,0]),axis=0)

        pred_field = pp.invert_normalize_function(pred_field, field_name, coefs_dict)
        real_field = pp.invert_normalize_function(real_field, field_name, coefs_dict)

        line_pred.set_ydata(pred_field)
        line_real.set_ydata(real_field)

        ax.set_title('{:.2f} s'.format(float(times[i])))
        return

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(pred_states),
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=framerate)
    anim.save(outfile_name, writer = writervideo)

def plot_3D(model_name, states, times,
            coefs_dict, field_name, outfile_name, time, framerate = 60):

    nframes = time * framerate
    indices = np.floor(np.linspace(0,len(states)-1,nframes)).astype(int)

    selected_times = []
    selected_states = []
    for ind in indices:
        selected_states.append(states[ind])
        selected_times.append(times[0,ind])

    states = selected_states
    times = selected_times

    graph = pp.load_graphs('../graphs/data/' + model_name + '.grph')[0][0]

    cmap = cm.get_cmap("viridis")
    fig, ax, points = plot_3D_graph(graph, states[0], coefs_dict, field_name, cmap)

    angles = np.floor(np.linspace(0,360,len(states))).astype(int)

    def animation_frame(i):
        ax.view_init(elev=10., azim=angles[i])
        state = states[i]
        fin = state[field_name]['inlet']
        fout = state[field_name]['outlet']
        finner = state[field_name]['inner']
        field = np.concatenate((fin,fout,finner),axis=0)
        field = pp.invert_normalize_function(field, field_name, coefs_dict)

        C = (field - coefs_dict[field_name]['min']) / \
            (coefs_dict[field_name]['max'] - coefs_dict[field_name]['min'])

        points.set_color(cmap(C))
        ax.set_title('{:.2f} s'.format(float(times[i])))
        return

    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=len(states),
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=framerate)
    anim.save(outfile_name, writer = writervideo)


model_name = '0063_1001'
print('Create geometry: ' + model_name)
soln = io.read_geo('../graphs/vtps/' + model_name + '.vtp').GetOutput()
fields, _, p_array = io.get_all_arrays(soln, None)

geometry = ResampledGeometry(Geometry(p_array), 5, remove_caps = True, doresample = True)

plot3Dgeo(geometry, fields['area'], 1, fields['area'], 0, 1)
