import sys
import os
sys.path.append(os.getcwd())
import tools.io_utils as io
import dgl
import torch as th
from tqdm import tqdm
from dgl.data.utils import load_graphs as lg
import numpy as np
import json
import random

def normalize(field, field_name, statistics, norm_dict_label):
    """
    Normalize field.

    Normalize a field using statistics provided as input.

    Arguments:
        field: the field to normalize
        field_name (string): name of field
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'features' or 'labels'
    Returns:
        normalized field

    """
    if statistics['normalization_type'][norm_dict_label] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        if np.abs(delta) > 1e-5:
            field = (field - statistics[field_name]['min']) / delta
        else:
            field = field * 0
    elif statistics['normalization_type'][norm_dict_label] == 'normal':
        delta = statistics[field_name]['stdv']
        if np.abs(delta) > 1e-5 and not np.isnan(delta):
            field = (field - statistics[field_name]['mean']) / delta
        else:
            field = field * 0
    elif statistics['normalization_type'][norm_dict_label] == 'none':
        pass
    else:
        raise Exception('Normalization type not implemented')
    return field

def invert_normalize(field, field_name, statistics, norm_dict_label):
    """
    Invert normalization over field.

    Invert normalization using statistics provided as input.

    Arguments:
        field: the field to normalize
        field_name (string): name of field
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'feature' or 'label'
    Returns:
        normalized field

    """
    if statistics['normalization_type'][norm_dict_label] == 'min_max':
        delta = (statistics[field_name]['max'] - statistics[field_name]['min'])
        field = statistics[field_name]['min'] + delta * field
    elif statistics['normalization_type'][norm_dict_label] == 'normal':
        delta = statistics[field_name]['stdv']
        if np.abs(delta) > 1e-5 and not np.isnan(delta):
            field = statistics[field_name]['mean'] + delta * field
        else:
            field = statistics[field_name]['mean']
    elif statistics['normalization_type'][norm_dict_label] == 'none':
        pass
    else:
        raise Exception('Normalization type not implemented')
    return field

def load_graphs(input_dir, n_graphs_to_keep = -1):
    """
    Load all graphs in directory.

    Load all graphs in input_dir.

    Arguments:
        input_dir (string): input directory path
        n_graphs_to_keep: number of graphs to keep. If -1, keep all graphs.
                          Default value -> -1.

    Returns:
        list of DGL graphs

    """
    files = os.listdir(input_dir)
    random.seed(10)
    random.shuffle(files)

    graphs = {}
    for file in tqdm(files, desc = 'Loading graphs', colour='green'):
        if 'grph' in file:
            graphs[file] = lg(input_dir + file)[0][0]

    return graphs

def compute_statistics(graphs, fields, statistics):
    """
    Compute statistics on a list of graphs.

    The computet statistics are: min value, max value, mean, and standard
    deviation.

    Arguments:
        graphs: list of graphs
        fields: dictionary containing field names, divided into node and edge
                fields
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
    Returns:
        dictionary containining statistics (key: statistics name, value: value).
        New fields are appended to the input 'statistics' argument.

    """
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

    graph_sts = {'nodes': [], 'edges': [], 'tsteps': []}

    for graph_n in graphs:
        graph = graphs[graph_n]
        graph_sts['nodes'].append(graph.ndata['x'].shape[0])
        graph_sts['edges'].append(graph.edata['distance'].shape[0])
        graph_sts['tsteps'].append(graph.ndata['pressure'].shape[2])

    for name in graph_sts:
        cur_statistics = {}

        cur_statistics['min'] = np.min(graph_sts[name])
        cur_statistics['max'] = np.max(graph_sts[name])
        cur_statistics['mean'] = np.mean(graph_sts[name])
        cur_statistics['stdv'] = np.std(graph_sts[name])

        statistics[name] = cur_statistics

    return statistics

def normalize_graphs(graphs, fields, statistics, norm_dict_label):
    """
    Normalize all graphs in a list.

    Arguments:
        graphs: list of graphs
        fields: dictionary containing field names, divided into node and edge
                fields
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'features' or 'labels'

    """
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
    """
    Add features to graphs.

    This function adds node and edge features to all graphs in
    the input list.

    Arguments:
        graphs: list of graphs
        params: dictionary containing parameters of the normalization

    """
    for graph_n in tqdm(graphs, desc = 'Add features', colour='green'):
        graph = graphs[graph_n]
        ntimes = graph.ndata['pressure'].shape[2]

        graph.ndata['dt'].repeat(1, 1, ntimes)
        area = graph.ndata['area'].repeat(1, 1, ntimes)
        tangent = graph.ndata['tangent'].repeat(1, 1, ntimes)
        type = graph.ndata['type'].repeat(1, 1, ntimes)

        p = graph.ndata['pressure'].clone()
        q = graph.ndata['flowrate'].clone()

        graph.ndata['nfeatures'] = th.cat((p, q, area, tangent,
                                           type), axis = 1)

        rp = graph.edata['rel_position']
        rpn = graph.edata['distance']
        if 'type' in graph.edata:
            rpt = graph.edata['type']
            graph.edata['efeatures'] = th.cat((rp, rpn, rpt), axis = 1)
        else:
            graph.edata['efeatures'] = th.cat((rp, rpn), axis = 1)

def add_deltas(graphs):
    """
    Compute pressure and flowrate increments.

    The increments are computed from time t to t+1 and stored as node features
    labelled 'dp' and 'dq'

    Arguments:
        graphs: list of graphs

    """
    for graph_n in tqdm(graphs, desc = 'Add deltas', colour='green'):
        graph = graphs[graph_n]

        graph.ndata['dp'] = graph.ndata['pressure'][:,:,1:] - \
                            graph.ndata['pressure'][:,:,:-1]

        graph.ndata['dq'] = graph.ndata['flowrate'][:,:,1:] - \
                            graph.ndata['flowrate'][:,:,:-1]

def save_graphs(graphs, output_dir):
    """
    Save all graphs contained in a list to file.

    Arguments:
        graphs: list of graphs
        output_dir: path of output directory

    """
    for graph_name in tqdm(graphs, desc = 'Saving graphs', colour='green'):
        dgl.save_graphs(output_dir + graph_name, graphs[graph_name])

def save_parameters(params, output_dir):
    """
    Save normalization parameters to file .

    Arguments:
        params: dictionary containing nornalization parameters
        output_dir: path of output directory

    """
    with open(output_dir + '/parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4)

def restrict_graphs(graphs, types, types_to_keep):
    """
    Restrict the list of graphs to the types that we are interested in.

    Arguments:
        graphs: list of graphs
        types: dictionary with all types (key: model name, value: type)
        types_to_keep: list with types to keep:
    
    Returns:
        restricted list of DGL graphs

    """
    selected_graphs = {}
    for graph in graphs:
        if types[graph.split('.')[0]] in types_to_keep:
            selected_graphs[graph] = graphs[graph]
    graphs = selected_graphs
    return graphs

def generate_normalized_graphs(input_dir, norm_type, bc_type,
                               types_to_keep = None,
                               n_graphs_to_keep = -1,
                               statistics = None):
    """
    Generate normalized graphs.

    Arguments:
        input_dir: path to input directory
        norm_type: dictionary with keys: features/labels,
                   values: min_max/normal
        bc_type: boundary condition type. Currently supported: full_dirichlet
                 (pressure and flowrate imposed at boundary nodes) and
                 realistic dirichlet (flowrate imposed at inlet, pressure
                 imposed at outlets)
        types_to_keep: dictionary containing all graphs types, and list
                       containing types we want to keep. If None, keep all
                       types. Default value -> None.
        n_graphs_to_keep: number of graphs to keep. If -1, keep all graphs.
                          Default value -> -1.

    Return:
        list of normalized graphs
        dictionary of parameters

    """
    fields_to_normalize = {'node': ['area', 'pressure',
                                'flowrate', 'dt'],
                       'edge': ['distance']}

    docompute_statistics = True
    if statistics != None:
        docompute_statistics = False

    if docompute_statistics:
        statistics = {'normalization_type': norm_type}
    graphs = load_graphs(input_dir, n_graphs_to_keep)

    if types_to_keep != None:
        graphs = restrict_graphs(graphs, types_to_keep['types'], 
                                types_to_keep['types_to_keep'])

    if n_graphs_to_keep != -1:
        graphs_ = {}
        count = 0
        for key, value in graphs.items():
            if count == n_graphs_to_keep:
                break
            graphs_[key] = value
            count = count + 1
        graphs = graphs_

    if docompute_statistics:
        compute_statistics(graphs, fields_to_normalize, statistics)
    normalize_graphs(graphs, fields_to_normalize, statistics, 'features')
    add_deltas(graphs)
    if docompute_statistics:
        compute_statistics(graphs, {'node' : ['dp', 'dq']}, statistics)
    normalize_graphs(graphs, {'node' : ['dp', 'dq']}, statistics, 'labels')
    params = {'bc_type': bc_type}
    params['statistics'] = statistics
    add_features(graphs, params)

    return graphs, params