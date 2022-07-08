import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import graph1d.generate_normalized_graphs as gng
from matplotlib import animation
import torch as th

def plot_history(history_train, history_test, label, folder):
    plt.figure()
    ax = plt.axes()

    ax.plot(history_train[0], history_train[1], color = 'red', label='train')
    ax.plot(history_test[0], history_test[1], color = 'blue', label='test')
    ax.legend()
    ax.set_xlim((history_train[0][0],history_train[0][-1]))

    ax.set_xlabel('epoch')
    ax.set_ylabel(label)
    ax.set_yscale('log')

    plt.savefig(folder + '/' + label + '.png')

def video_all_nodes(features, graph, params, time, 
                    outfile_name, framerate = 60):
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

                                 
                                   