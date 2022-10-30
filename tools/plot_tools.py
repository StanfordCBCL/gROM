import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import graph1d.generate_normalized_graphs as gng
import matplotlib
from matplotlib import animation
import torch as th
import os
import plotly.graph_objects as go
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from random import sample
import graph1d.generate_normalized_graphs as nz
import matplotlib.cm as cm
# from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import matplotlib.ticker as ticker

# some colors
Cardinal_red = "#8F353C"
Cardinal_blue = "#54A0C0"
Gray = "#c5c7c5"
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [Cardinal_red, Cardinal_blue, CB91_Violet, CB91_Green, CB91_Pink,
              CB91_Amber, CB91_Purple]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


def plot_graph(points, bif_id, indices, edges1, edges2, 
               stl_mesh = None, linewidth = 0.3, s = 1):
    """
    Plot a graph along with the corresponding stl (if available)

    Arguments:
        points: 3D array with all graph nodes
        bif_id: 1D array containing bifurcation id (-1 if a node is a branch
                node)
        indices: dictionary containing lists of inlet and outlet nodes
        edges1: 1D array containing indices of first nodes in edges
        edges2: 1D array containing indices of second nodes in edges
        stl_mesh: if not None, plots a semi-transparent geometry along with
                  the graph
        linewidth: linewidth for edges. Default -> 0.3
        s: parameter controlling point sizes. Default -> 1

    """
    plt.figure()
    ax = plt.axes(projection='3d')
    ax._axis3don = False

    np.min(bif_id)
    np.max(bif_id)

    branch_nodes = np.where(bif_id == -1)[0]
    branch_nodes = branch_nodes[1:]
    jun_nodes = np.where(bif_id > -1)[0]

    ax.scatter(points[branch_nodes,0], 
               points[branch_nodes,1], 
               points[branch_nodes,2], 
               color = CB91_Blue,
               depthshade=0, s = s)

    ax.scatter(points[jun_nodes,0], 
            points[jun_nodes,1], 
            points[jun_nodes,2], 
            color = CB91_Amber,
            depthshade=0, s = s)

    inlet = indices['inlet']
    ax.scatter(points[inlet,0], points[inlet,1], points[inlet,2],               color='green', depthshade=0, s = s * 10)

    outlets = indices['outlets']
    ax.scatter(points[outlets,0], points[outlets,1], points[outlets,2],color='red', depthshade=0, s = s * 10)

    for iedge in range(edges1.size):
        ax.plot3D([points[edges1[iedge],0],points[edges2[iedge],0]],
                  [points[edges1[iedge],1],points[edges2[iedge],1]],
                  [points[edges1[iedge],2],points[edges2[iedge],2]],
                   color = 'black', linewidth = linewidth, alpha = 0.5)

    ax.set_box_aspect((np.ptp(points[:,0]), 
                       np.ptp(points[:,1]), 
                       np.ptp(points[:,2])))
    if stl_mesh != None:
        ax.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors,
                            alpha=0.08))
    plt.box(False)

def plot_history(history_train, history_test, label, folder = None):
    """
    Plot a graph along with the corresponding stl (if available)

    Arguments:
        history_train: list of train metrics. First value: epochs. Second value:
                        metric
        history_test: list of test metrics. First value: epochs. Second value:
                        metric
        label (string): name of the metric to use for y-axis and plot title
        folder: output folder. Default -> None

    """
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.set_aspect('auto')
    ax.plot(history_train[0], history_train[1], linewidth = 3, label='train')
    ax.plot(history_test[0], history_test[1], linewidth = 3, label='test')
    ax.legend()
    ax.set_xlim((history_train[0][0],history_train[0][-1]))

    ax.set_xlabel('epoch')
    ax.set_ylabel(label)
    plt.tight_layout()

    plt.legend(frameon=False)

    if folder != None:
        plt.savefig(folder + '/' + label + '.eps')
    else:
        plt.show()

def video_all_nodes(features, graph, params, time, 
                    outfile_name, framerate = 60):
    """
    Creates and saves a .mp4 video with pressure anf flow rate values for 
    all nodes in the graph.

    Arguments:
        features: 3D array with reconstructed features
        graph: the GNN graph
        params: dictionary of parameters
        time: duration of the video in seconds
        outfile_name (string): name of the output video
        framerate (int): framerate. Default -> 60

    """
    nframes = time * framerate

    indices = np.floor(np.linspace(0,features.shape[2]-1,nframes)).astype(int)

    sel_pred_features = features[:,:,indices]
    sel_real_features = graph.ndata['nfeatures'][:,:,indices]

    sel_pred_features[:,0,:] = gng.invert_normalize(sel_pred_features[:,0,:],
                                                   'pressure',
                                                   params['statistics'],
                                                   'features')
    sel_pred_features[:,1,:] = gng.invert_normalize(sel_pred_features[:,1,:],
                                                'flowrate',
                                                params['statistics'],
                                                'features')
    sel_real_features[:,0,:] = gng.invert_normalize(sel_real_features[:,0,:],
                                                   'pressure',
                                                   params['statistics'],
                                                   'features')
    sel_real_features[:,1,:] = gng.invert_normalize(sel_real_features[:,1,:],
                                                'flowrate',
                                                params['statistics'],
                                                'features')
    minp = th.min(sel_real_features[:,0,:])
    maxp = th.max(sel_real_features[:,0,:])
    minq = th.min(sel_real_features[:,1,:])
    maxq = th.max(sel_real_features[:,1,:])

    fig, ax = plt.subplots(2, dpi = 284)

    nodes = np.arange(features.shape[0])
    scatter_real_p = ax[0].scatter(nodes, sel_real_features[:,0,0], 
                                   color = 'black', 
                                   s = 1.5, alpha = 0.3)
    scatter_pred_p = ax[0].scatter(nodes, sel_pred_features[:,0,0], 
                                   color = 'red', s = 1.5, alpha = 1)
    scatter_real_q = ax[1].scatter(nodes, sel_real_features[:,1,0], 
                                   color = 'black', s = 1.5, alpha = 0.3)
    scatter_pred_q = ax[1].scatter(nodes, sel_pred_features[:,1,0], 
                                   color = 'red', s = 1.5, alpha = 1)
    nodesidxs = np.expand_dims(nodes, axis = 1)
    ax[1].set_xlabel('graph node index')
    ax[0].set_ylabel('pressure [mmHg]')
    ax[1].set_ylabel('flowrate [cm^3/s]')
    def animation_frame(i):
        p = sel_real_features[:,0,i]
        p = np.concatenate((nodesidxs, np.expand_dims(p, axis = 1)),axis = 1)
        scatter_real_p.set_offsets(p)
        p = sel_pred_features[:,0,i]
        p = np.concatenate((nodesidxs, np.expand_dims(p, axis = 1)),axis = 1)
        scatter_pred_p.set_offsets(p)
        q = sel_real_features[:,1,i]
        q = np.concatenate((nodesidxs, np.expand_dims(q, axis = 1)),axis = 1)
        scatter_real_q.set_offsets(q)
        q = sel_pred_features[:,1,i]
        q = np.concatenate((nodesidxs, np.expand_dims(q, axis = 1)),axis = 1)
        scatter_pred_q.set_offsets(q)

        # ax[0].set_title('{:.2f} s'.format(float(times[i])))
        ax[0].set_xlim(0,features.shape[0])
        ax[0].set_ylim((minp, maxp))
        ax[1].set_xlim(0,features.shape[0])
        ax[1].set_ylim((minq, maxq))
 
        return scatter_pred_p,
    
    anim = animation.FuncAnimation(fig, animation_frame,
                                   frames=indices.size,
                                   interval=20)
    writervideo = animation.FFMpegWriter(fps=framerate)
    anim.save(outfile_name, writer = writervideo)
