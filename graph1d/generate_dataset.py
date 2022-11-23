import sys
import os
from tkinter import XView
sys.path.append(os.getcwd())
import tools.io_utils as io
from dgl.data import DGLDataset
import time
import graph1d.generate_normalized_graphs as nz
import random
import numpy as np
import copy
import torch as th
from tqdm import tqdm

nchunks = 10

class Dataset(DGLDataset):
    """
    Class to store and traverse a DGL dataset.

    Attributes:
        graphs: list of graphs in the dataset
        params: dictionary containing parameters of the problem
        times: array containing number of times for each graph in the dataset
        lightgraphs: list of graphs, without edge and node features
        graph_names: n x 2 array (n is the total number of timesteps in the
                     dataset) mapping a graph index (first column) to the
                     timestep index (second column).

    """
    def __init__(self, graphs, params, graph_names):
        """
        Init Dataset.

        Init Dataset with list of graphs, dictionary of parameters, and list of
        graph names.

        Arguments:
            graphs: lift of graphs
            params: dictionary of parameters
            graph_names: list of graph names
            index_map:

        """
        self.graphs = graphs
        self.params = params
        self.times = []
        self.lightgraphs = []
        self.graph_names = graph_names
        super().__init__(name='dataset')

    def create_index_map(self):
        """
        Create index map.

        Index map is a n x 2 array (n is the total number of timesteps in the
        dataset) mapping a graph index (first column) to the timestep index
        (second column).

        """
        i = 0
        offset = 0
        ngraphs = len(self.times)
        stride = self.params['stride']
        self.index_map = np.zeros((self.total_times - stride * ngraphs, 2))
        for t in self.times:
            # actual time (minus stride)
            at = t - stride
            graph_index = np.ones((at, 1)) * i
            time_index = np.expand_dims(np.arange(0, at), axis = 1)
            self.index_map[offset:at + offset,:] = np.concatenate((graph_index,
                                                                   time_index),
                                                                   axis = 1)
            i = i + 1
            offset = offset + at
        self.index_map = np.array(self.index_map, dtype = int)

    def process(self):
        """
        Process Dataset.

        This function creates lightgraphs, the index map, and collects all times
        from the graphs.

        """
        start = time.time()

        for graph in tqdm(self.graphs, desc = 'Processing dataset',
                          colour='green'):

            lightgraph = copy.deepcopy(graph)

            node_data = [ndata for ndata in lightgraph.ndata]
            edge_data = [edata for edata in lightgraph.edata]
            for ndata in node_data:
                if 'mask' not in ndata:
                    del lightgraph.ndata[ndata]
            for edata in edge_data:
                del lightgraph.edata[edata]

            self.times.append(graph.ndata['nfeatures'].shape[2])
            self.lightgraphs.append(lightgraph)

        self.times = np.array(self.times)
        self.total_times = np.sum(self.times)

        self.create_index_map()

        end = time.time()
        elapsed_time = end - start
        print('\tDataset generated in {:0.2f} s'.format(elapsed_time))

    def get_lightgraph(self, i):
        """
        Get ith lightgraph

        Noise is added to node features of the graph (pressure and flowrate).

        Arguments:
            i: index of the graph

        Returns:
            The DGL graph
        """
        indices = self.index_map[i,:]
        igraph = indices[0]
        itime = indices[1]

        features = self.graphs[igraph].ndata['nfeatures'].clone()

        nf = features[:,:,itime].clone()
        nfsize = nf[:,:2].shape

        dt = nz.invert_normalize(self.graphs[igraph].ndata['dt'][0], 'dt',
                                 self.params['statistics'], 'features')

        curnoise = np.random.normal(0, self.params['rate_noise'] * dt, nfsize)
        nf[:,:2] = nf[:,:2] + curnoise

        fnoise = np.random.normal(0, self.params['rate_noise_features'],
                                  nf[:,2:].shape)
        # flowrate at inlet is exact
        fnoise[self.graphs[igraph].ndata['inlet_mask'].bool(),1] = 0
        nf[:,2:] = nf[:,2:] + fnoise

        self.lightgraphs[igraph].ndata['nfeatures'] = nf

        gf = self.graphs[igraph].ndata['geom_features']
        self.lightgraphs[igraph].ndata['geom_features'] = gf

        ns = features[:,0:2,itime + 1:itime + 1 + self.params['stride']].clone()

        self.lightgraphs[igraph].ndata['next_steps'] = ns

        ef = self.graphs[igraph].edata['efeatures']

        # add regular noise to the edge features to prevent overfitting
        fnoise = np.random.normal(0, self.params['rate_noise_features'],
                                  ef[:,2:].shape)
        ef[:,2:] = ef[:,2:] + fnoise
        self.lightgraphs[igraph].edata['efeatures'] = ef.squeeze()

        return self.lightgraphs[igraph]

    def __getitem__(self, i):
        """
        Get ith lightgraph

        Arguments:
            i: index of the lightgraph

        Returns:
            ith lightgraph
        """
        return self.get_lightgraph(i)

    def __len__(self):
        """
        Length of the dataset

        Length of the dataset is the total number of timesteps (minus stride).

        Returns:
            length of the Dataset
        """
        return self.index_map.shape[0]

    def __str__(self):
        """
        Returns graph names.

        Returns:
            graph names
        """
        print('Total number of graphs: {:}'.format(self.__len__()))
        return 'Dataset = ' + ', '.join(self.graph_names)

def split(graphs, divs, types):
    """
    Split a list of graphs.

    The graphs are split into multiple train/test groups. Number of groups is
    determined by the divs argument. The function takes as input the type of
    graphs to make the datasets balanced.

    Arguments:
        divs: number of train/test groups.
        types: dictionary (key: graph name, value: type)

    Returns:
        List of groups
    """
    def chunks(lst, n):
        retlist = [[] for l in range(n)]
        for i, el in enumerate(lst):
            retlist[i % n].append(el)
        return retlist

    names = [graph_name for graph_name in graphs]

    if len(names) == 1:
        datasets = [{'train': [names[0]],
                     'test': [names[0]]}]
        return datasets

    # the seed MUST be set when using parallelism! Otherwise cores get different
    # splits
    random.seed(10)
    random.shuffle(names)

    sublists = {}
    for name in names:
        type = types[name.split('.')[0]]
        if type not in sublists:
            sublists[type] = []
        sublists[type].append(name)

    subsets = {}
    for sublist_n, sublist_v in sublists.items():
        subsets[sublist_n] = list(chunks(sublist_v, divs))
        nsets = len(subsets[sublist_n])
        # we distribute the last sets among the first n-1
        if nsets != divs:
            for i, graph in enumerate(subsets[sublist_n][-1]):
                subsets[sublist_n][i % divs] += [graph]
            del subsets[sublist_n][-1]
        nsets = len(subsets[sublist_n])

    datasets = []

    for i in range(1):
        cur_set = []
        for _, subset_v in subsets.items():
            cur_set = cur_set + subset_v[i]

        newdata = {'test': cur_set}
        train_s = []
        for j in range(nsets):
            if j != i:
                cur_set = []
                for _, subset_v in subsets.items():
                    cur_set = cur_set + subset_v[j]
                train_s = train_s + cur_set
        newdata['train'] = train_s
        datasets.append(newdata)

    return datasets

def generate_dataset(graphs, params, types):
    """
    Generate a list of datasets

    The returned list is composed of dictionary containing train and test
    Datasets. The function takes as input the type of graphs to make the
    datasets balanced.

    Arguments:
        graphs: list of graphs
        params: dictionary of parameters
        types: dictionary (key: graph name, value: type)

    Returns:
        List of datasets
    """
    dataset_list = []
    datasets = split(graphs, nchunks, types)
    for dataset in datasets:
        train_graphs = [graphs[gname] for gname in dataset['train']]
        train_dataset = Dataset(train_graphs, params, dataset['train'])

        test_graphs = [graphs[gname] for gname in dataset['test']]
        test_dataset = Dataset(test_graphs, params, dataset['test'])

        dataset_list.append({'train': train_dataset, 'test': test_dataset})

    print('Generated {:} datasets'.format(len(dataset_list)))
    for dataset in dataset_list:
        print('Train size = {:}'.format(len(dataset['train'].graph_names)))
        print('Test size = {:}'.format(len(dataset['test'].graph_names)))

    return dataset_list

def generate_dataset_from_params(graphs, params):
    """
    Generate a dataset from parameters

    The dictionary of parameters must contain information about train-test
    split.

    Arguments:
        graphs: list of graphs
        params: dictionary of parameters

    Returns:
        Dataset

    """

    train = [graphs[train_graph] for train_graph in params['train_split']]
    test = [graphs[test_graph] for test_graph in params['test_split']]

    train_dataset = Dataset(train, params, params['train_split'])
    test_dataset = Dataset(test, params, params['test_split'])

    dataset = {'train': train_dataset, 'test': test_dataset}

    print('Train size = {:}'.format(len(dataset['train'].graph_names)))
    print('Test size = {:}'.format(len(dataset['test'].graph_names)))

    return dataset
