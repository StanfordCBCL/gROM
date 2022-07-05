import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("../tools/")

import io_utils as io
from dgl.data import DGLDataset
import time
import generate_normalized_graphs as nz
import random
import numpy as np
import copy
import torch as th
from tqdm import tqdm

params = {}
nchuncks = 10

class Dataset(DGLDataset):
    def __init__(self, graphs, params):
        self.graphs = graphs
        self.params = params
        self.times = []
        self.lightgraphs = []
        super().__init__(name='dataset')

    def create_index_map(self):
        i = 0
        offset = 0
        self.index_map = np.zeros((self.total_times, 2))
        for t in self.times:
            graph_index = np.ones((t, 1)) * i
            time_index = np.expand_dims(np.arange(0, t), axis = 1)
            self.index_map[offset:t + offset,:] = np.concatenate((graph_index, 
                                                                  time_index),
                                                                  axis = 1)
            i = i + 1
            offset = offset + t
        self.index_map = np.array(self.index_map, dtype = int)
            
    def process(self):
        start = time.time()

        for graph in tqdm(self.graphs, desc = 'Processing dataset', 
                          colour='green'):

            lightgraph = copy.deepcopy(graph)

            node_data = [ndata for ndata in lightgraph.ndata]
            edge_data = [edata for edata in lightgraph.edata]
            for ndata in node_data:
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
        indices = self.index_map[i,:]


        nf = self.graphs[indices[0]].ndata['nfeatures'][:,:,indices[1]].clone()
        nfsize = nf[:,:2].shape
        curnoise = np.random.normal(0, self.noise_rate, nfsize)
        nf[:,:2] = nf[:,:2] + curnoise
        self.lightgraphs[indices[0]].ndata['nfeatures'] = nf

        nl = self.graphs[indices[0]].ndata['nlabels'][:,:,indices[1]].clone()
        nl[:,:2] = nl[:,:2] - curnoise
        self.lightgraphs[indices[0]].ndata['nlabels'] = nl

        ef = self.graphs[indices[0]].edata['efeatures']
        self.lightgraphs[indices[0]].edata['efeatures'] = ef.squeeze()

        return self.lightgraphs[indices[0]]

    def set_noise_rate(self, noise_rate):
        self.noise_rate = noise_rate
        # self.noises = []
        # for graph in self.graphs:
        #     nsize = graph.ndata['nfeatures'][:,:2,:].shape
        #     self.noises.append(np.random.normal(0, noise_rate, nsize))

    def __getitem__(self, i):
        return self.get_lightgraph(i)

    def __len__(self):
        return self.total_times

def split(graphs, divs):
    def chunks(lst, n):
        n = int(np.floor(len(lst)/n))
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    names = [graph_name for graph_name in graphs]

    if len(names) == 1:
        datasets = [{'train': [names[0]],
                     'test': [names[0]]}]
        return datasets

    random.seed(10)
    random.shuffle(names)

    sets = list(chunks(names, divs))
    nsets = len(sets)

    datasets = []    

    for i in range(1):
        newdata = {'test': sets[i]}
        train_s = []
        for j in range(nsets):
            if j != i:
                train_s = train_s + sets[j]
        newdata['train'] = train_s
        datasets.append(newdata)

    return datasets

def generate_dataset(input_dir):

    graphs = nz.load_all_graphs(input_dir)

    dataset_list = []
    datasets = split(graphs, nchuncks)    
    for dataset in datasets:
        train_graphs = [graphs[gname] for gname in dataset['train']]
        train_dataset = Dataset(train_graphs, params)

        test_graphs = [graphs[gname] for gname in dataset['test']]
        test_dataset = Dataset(test_graphs, params)

        dataset_list.append({'train': train_dataset, 'test': test_dataset})

    return dataset_list