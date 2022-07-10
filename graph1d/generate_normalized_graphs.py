import sys
import os
sys.path.append(os.getcwd())
import tools.io_utils as io
import dgl
import torch as th
from tqdm import tqdm
from dgl.data.utils import load_graphs
import numpy as np
import json

def normalize(field, field_name, statistics, norm_dict_label):
    if statistics['normalization_type'][norm_dict_label] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        if np.abs(delta) > 1e-8:
            field = (field - statistics[field_name]['min']) / delta
        else:
            field = field * 0
    elif statistics['normalization_type'][norm_dict_label] == 'normal':
        delta = statistics[field_name]['stdv']
        if np.abs(delta) > 1e-8:
            field = (field - statistics[field_name]['mean']) / delta
        else:
            field = field * 0
    elif statistics['normalization_type'][norm_dict_label] == 'none':
        pass
    else:
        raise Exception('Normalization type not implemented')
    return field

def invert_normalize(field, field_name, statistics, norm_dict_label):
    if statistics['normalization_type'][norm_dict_label] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        field = statistics[field_name]['min'] + delta * field
    elif statistics['normalization_type'][norm_dict_label] == 'normal':
        delta = statistics[field_name]['stdv']
        field = statistics[field_name]['mean'] + delta * field
    elif statistics['normalization_type'][norm_dict_label] == 'none':
        pass
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

                    if field_name == 'area':
                        print(graph_n)
                        print(minv)
                        print(maxv)

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

def normalize_graphs(graphs, fields, statistics, norm_dict_label):
    print('Normalize graphs')
    for etype in fields:
            for field_name in fields[etype]:
                for graph_n in tqdm(graphs, desc = field_name, 
                                    colour='green'):
                    graph = graphs[graph_n]
                    if etype == 'node':
                        d = graph.ndata[field_name]
                        graph.ndata[field_name] = normalize(d, field_name,
                                                            statistics,
                                                            norm_dict_label)    
                                
                    if etype == 'edge':
                        d = graph.edata[field_name]
                        graph.edata[field_name] = normalize(d, field_name,
                                                            statistics,
                                                            norm_dict_label)

def add_features(graphs, params):  
    for graph_n in tqdm(graphs, desc = 'Add features', colour='green'):
        graph = graphs[graph_n]
        ntimes = graph.ndata['dp'].shape[2]

        dt = graph.ndata['dt'].repeat(1, 1, ntimes)
        area = graph.ndata['area'].repeat(1, 1, ntimes)
        type = graph.ndata['type'].repeat(1, 1, ntimes)

        p = graph.ndata['pressure'][:,:,:-1].clone()
        q = graph.ndata['flowrate'][:,:,:-1].clone()

        dp = graph.ndata['dp'].clone()
        dq = graph.ndata['dq'].clone()

        # set boundary conditions
        if params['bc_type'] == 'realistic_dirichlet':
            p[graph.ndata['outlet_mask'].bool(),:,:] = \
                graph.ndata['pressure'][graph.ndata['outlet_mask'].bool(),:,1:]
            q[graph.ndata['inlet_mask'].bool(),:,:] = \
                graph.ndata['flowrate'][graph.ndata['inlet_mask'].bool(),:,1:]
            # mask out labels at boundary nodes
            dp[graph.ndata['outlet_mask'].bool(),:,:] = 0
            dq[graph.ndata['inlet_mask'].bool(),:,:] = 0
        elif params['bc_type'] == 'full_dirichlet':
            p[graph.ndata['inlet_mask'].bool(),:,:] = \
                graph.ndata['pressure'][graph.ndata['inlet_mask'].bool(),:,1:]
            p[graph.ndata['outlet_mask'].bool(),:,:] = \
                graph.ndata['pressure'][graph.ndata['outlet_mask'].bool(),:,1:]
            q[graph.ndata['inlet_mask'].bool(),:,:] = \
                graph.ndata['flowrate'][graph.ndata['inlet_mask'].bool(),:,1:]
            q[graph.ndata['outlet_mask'].bool(),:,:] = \
                graph.ndata['flowrate'][graph.ndata['outlet_mask'].bool(),:,1:]
            dp[graph.ndata['inlet_mask'].bool(),:,:] = 0
            dp[graph.ndata['outlet_mask'].bool(),:,:] = 0
            dq[graph.ndata['inlet_mask'].bool(),:,:] = 0
            dq[graph.ndata['outlet_mask'].bool(),:,:] = 0

        graph.ndata['nfeatures'] = th.cat((p, q, area, type, dt), axis = 1)
        graph.ndata['nlabels'] = th.cat((dp, dq), axis = 1)

        rp = graph.edata['rel_position']
        rpn = graph.edata['distance']
        if 'type' in graph.edata:
            rpt = graph.edata['type']
            graph.edata['efeatures'] = th.cat((rp, rpn, rpt), axis = 1)
        else:
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

def save_parameters(params, output_dir):
    with open(output_dir + '/parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4)

def generate_normalized_graphs(input_dir, norm_type, bc_type):
    fields_to_normalize = {'node': ['area', 'pressure', 
                                'flowrate', 'dt'], 
                       'edge': ['distance']}
    statistics = {'normalization_type': norm_type}
    graphs = load_all_graphs(input_dir)
    compute_statistics(graphs, fields_to_normalize, statistics)
    normalize_graphs(graphs, fields_to_normalize, statistics, 'features')
    add_deltas(graphs)
    compute_statistics(graphs, {'node' : ['dp', 'dq']}, statistics)
    normalize_graphs(graphs, {'node' : ['dp', 'dq']}, statistics, 'labels')
    params = {'bc_type': bc_type}
    params['statistics'] = statistics
    add_features(graphs, params)
    
    return graphs, params
    
if __name__ == "__main__":
    data_location = io.data_location()
    norm_type_features = 'normal'
    norm_type_labels = 'min_max'

    norm_type = {'features': norm_type_features, 'labels': norm_type_labels}
    graphs, params = generate_normalized_graphs(data_location + '/graphs/',
                                                norm_type, 'full_dirichlet')
    save_graphs(graphs, data_location + '/normalized_graphs/')
    save_parameters(params, data_location + '/normalized_graphs/')