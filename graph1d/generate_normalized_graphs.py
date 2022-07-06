import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("../tools/")

import io_utils as io
import dgl
import torch as th
from tqdm import tqdm
from dgl.data.utils import load_graphs
import numpy as np
import json

fields_to_normalize = {'node': ['area', 'pressure', 
                                'flowrate', 'dt'], 
                       'edge': ['rel_position_norm']}
normalization_type = 'normal'
# normalization_type = 'min_max'

def normalize(field, field_name, statistics):
    if statistics['normalization_type'] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        if np.abs(delta) > 1e-8:
            field = (field - statistics[field_name]['min']) / delta
        else:
            field = field * 0
    elif statistics['normalization_type'] == 'normal':
        delta = statistics[field_name]['stdv']
        if np.abs(delta) > 1e-8:
            field = (field - statistics[field_name]['mean']) / delta
        else:
            field = field * 0
    else:
        raise Exception('Normalization type not implemented')
    return field

def invert_normalize(field, field_name, statistics):
    if statistics['normalization_type'] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        field = statistics[field_name]['min'] + delta * field
    elif statistics['normalization_type'] == 'normal':
        delta = statistics[field_name]['stdv']
        field = statistics[field_name]['mean'] + delta * field
    else:
        raise Exception('Normalization type not implemented')
    return field

def load_all_graphs(input_dir):
    files = os.listdir(input_dir) 

    graphs = {}
    for file in tqdm(files, desc = 'Loading graphs', colour='green'):
        if 'grph' in file:
            graphs[file] = load_graphs(input_dir + file)[0][0]

    return graphs

def compute_statistics(graphs, fields, statistics):
    print('Compute statistics')
    for etype in fields:
            for field_name in fields[etype]:
                cur_statistics = {}
                minv = np.infty
                maxv = np.NINF
                Ns = []
                Ms = []
                means = []
                meansqs = []
                for graph_n in tqdm(graphs, desc = field_name, \
                                    colour='green'):
                    graph = graphs[graph_n]
                    if etype == 'node':
                        d = graph.ndata[field_name]
                    if etype == 'edge':
                        d = graph.edata[field_name]

                    # number of nodes
                    N = d.shape[0]
                    # number of times
                    M = d.shape[2]
                    minv = np.min([minv, th.min(d)])
                    maxv = np.max([maxv, th.max(d)])
                    mean = th.mean(d)
                    meansq = th.mean(d**2)

                    means.append(mean)
                    meansqs.append(meansq)
                    Ns.append(N)
                    Ms.append(M)

                ngraphs = len(graphs)
                MNs = 0
                for i in range(ngraphs):
                    MNs = MNs + Ms[i] * Ns[i]

                mean = 0
                meansq = 0
                for i in range(ngraphs):
                    coeff = Ms[i] * Ns[i] / MNs
                    mean = mean + coeff * means[i]
                    meansq = meansq + coeff * meansqs[i]

                cur_statistics['min'] = minv
                cur_statistics['max'] = maxv
                cur_statistics['mean'] = float(mean)
                cur_statistics['stdv'] = float(np.sqrt(meansq - mean**2))
                statistics[field_name] = cur_statistics
    return statistics

def normalize_graphs(graphs, fields, statistics):
    print('Normalize graphs')
    for etype in fields:
            for field_name in fields[etype]:
                for graph_n in tqdm(graphs, desc = field_name, 
                                    colour='green'):
                    graph = graphs[graph_n]
                    if etype == 'node':
                        d = graph.ndata[field_name]
                        graph.ndata[field_name] = normalize(d, field_name,
                                                            statistics)    
                                
                    if etype == 'edge':
                        d = graph.edata[field_name]
                        graph.edata[field_name] = normalize(d, field_name,
                                                            statistics)

def add_features(graphs):  
    for graph_n in tqdm(graphs, desc = 'Add features', colour='green'):
        graph = graphs[graph_n]
        ntimes = graph.ndata['dp'].shape[2]

        dt = graph.ndata['dt'].repeat(1, 1, ntimes)
        area = graph.ndata['area'].repeat(1, 1, ntimes)
        type = graph.ndata['type'].repeat(1, 1, ntimes)

        p = graph.ndata['pressure'][:,:,:-1].clone()
        q = graph.ndata['flowrate'][:,:,:-1].clone()
        # set boundary conditions
        p[graph.ndata['outlet_mask'].bool(),:,:] = \
                graph.ndata['pressure'][graph.ndata['outlet_mask'].bool(),:,1:]
        q[graph.ndata['inlet_mask'].bool(),:,:] = \
                graph.ndata['flowrate'][graph.ndata['inlet_mask'].bool(),:,1:]

        graph.ndata['nfeatures'] = th.cat((p, q, area, type, dt), axis = 1)

        dp = graph.ndata['dp']
        dq = graph.ndata['dq']
        # mask out labels at boundary nodes
        dp[graph.ndata['outlet_mask'].bool(),:,:] = 0
        dq[graph.ndata['inlet_mask'].bool(),:,:] = 0
        graph.ndata['nlabels'] = th.cat((dp, dq), axis = 1)

        rp = graph.edata['rel_position']
        rpn = graph.edata['rel_position_norm']
        graph.edata['efeatures'] = th.cat((rp, rpn), axis = 1)

def add_deltas(graphs):
    for graph_n in tqdm(graphs, desc = 'Add deltas', colour='green'):
        graph = graphs[graph_n]

        graph.ndata['dp'] = graph.ndata['pressure'][:,:,1:] - \
                            graph.ndata['pressure'][:,:,:-1]

        graph.ndata['dq'] = graph.ndata['flowrate'][:,:,1:] - \
                            graph.ndata['flowrate'][:,:,:-1]

def save_graphs(graphs, output_dir):
    for graph_name in tqdm(graphs, desc = 'Saving graphs', colour='green'):
        dgl.save_graphs(output_dir + graph_name, graphs[graph_name])

def save_statistics(statistics, output_dir):
    with open(output_dir + '/statistics.json', 'w') as outfile:
        json.dump(statistics, outfile, indent=4)
    
if __name__ == "__main__":
    data_location = io.data_location()
    input_dir = data_location + 'graphs/'
    output_dir = data_location + 'normalized_graphs/'

    statistics = {'normalization_type': normalization_type}
    graphs = load_all_graphs(input_dir)
    compute_statistics(graphs, fields_to_normalize, statistics)
    normalize_graphs(graphs, fields_to_normalize, statistics)
    add_deltas(graphs)
    compute_statistics(graphs, {'node' : ['dp', 'dq']}, statistics)
    normalize_graphs(graphs, {'node' : ['dp', 'dq']}, statistics)
    print(statistics)
    add_features(graphs)
    save_graphs(graphs, output_dir)
    save_statistics(statistics, output_dir)