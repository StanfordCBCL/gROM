from ast import Num
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
import pathlib
import tools.plot_tools as pt
import matplotlib.pyplot as plt
import generate_graphs as gg
from stl import mesh
from mpl_toolkits import mplot3d

if __name__ == "__main__":
    data_location = io.data_location()
    input_dir = data_location + 'vtps_aortas'
    input_dir_mesh = data_location + 'stls/'
    output_dir = data_location + 'graphs/'

    file = '0003_0001.1' # sys.argv[1]
    stl_mesh = None
    if os.path.exists(input_dir_mesh + file + '.stl'):
        stl_mesh = mesh.Mesh.from_file(input_dir_mesh + file + '.stl')
    point_data, points, edges1, edges2 = gg.load_vtp(file + '.vtp', input_dir)

    point_data['tangent'] = gg.generate_tangents(points, 
                                                 point_data['BranchIdTmp'])

    inlet = [0]
    outlets = gg.find_outlets(edges1, edges2)

    indices = {'inlet': inlet,
            'outlets': outlets}

    resample_perc = 0.03
    success = False
    while not success:
        try:
            sampled_indices, points, \
            edges1, edges2, _ = gg.resample_points(points.copy(),  
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
    outlets = gg.find_outlets(edges1, edges2)

    indices = {'inlet': inlet,
            'outlets': outlets}

    sampling_indices = np.arange(points.shape[0])
    partitions = [{'point_data': point_data,
                'points': points,
                'edges1': edges1,
                'edges2': edges2,
                'sampling_indices': sampling_indices}]

    for i, part in enumerate(partitions):
        filename = file.replace('.vtp','.' + str(i) + '.grph')
        add_boundary_edges = True
        add_junction_edges = True
        try:
            graph, indices, \
            points, bif_id, indices, \
            edges1, edges2 = gg.generate_graph(part['point_data'],
                                            part['points'],
                                            part['edges1'], 
                                            part['edges2'], 
                                            add_boundary_edges,
                                            add_junction_edges)
            pathlib.Path('images').mkdir(parents=True, exist_ok=True)
            pt.plot_graph(points, bif_id, indices, edges1, edges2, stl_mesh)
            plt.savefig('graph.png', 
                        format='png',
                        bbox_inches='tight',
                        dpi=1200)

        except Exception as e:
            print(e)                
    
