from base64 import encode
import torch
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import dgl.function as fn

class MLP(Module):
    def __init__(self, in_feats, out_feats, latent_space, n_h_layers, 
                normalize = True):
        super().__init__()
        self.input = Linear(in_feats,latent_space,bias = True).float()
        self.output = Linear(latent_space, out_feats, bias = True).float()
        self.n_h_layers = n_h_layers
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(self.n_h_layers):
            self.hidden_layers.append(Linear(latent_space, 
                                             latent_space, 
                                             bias = True).float())

        self.normalize = normalize
        if self.normalize:
            self.norm = LayerNorm(out_feats).float()

    def forward(self, inp):
        f = self.input(inp)
        f = F.leaky_relu(f)

        for i in range(self.n_h_layers):
            f = self.hidden_layers[i](f)
            f = F.leaky_relu(f)

        # enc_features = self.dropout(enc_features)
        f = self.output(f)

        if self.normalize:
            f = self.norm(f)

        return f

class MeshGraphNet(Module):
    def __init__(self, params):
        super(MeshGraphNet, self).__init__()

        self.encoder_nodes = MLP(8, 
                                 params['latent_size_gnn'],
                                 params['latent_size_mlp'],
                                 params['number_hidden_layers_mlp'])
        self.encoder_edges = MLP(4, 
                                 params['latent_size_gnn'],
                                 params['latent_size_mlp'],
                                 params['number_hidden_layers_mlp'])


        self.processor_nodes = torch.nn.ModuleList()
        self.processor_edges = torch.nn.ModuleList()
        self.process_iters = params['process_iterations']
        for i in range(self.process_iters):
            def generate_proc_MLP(in_feat):
                return MLP(in_feat,
                           params['latent_size_gnn'],
                           params['latent_size_mlp'],
                           params['number_hidden_layers_mlp'])

            lsgnn = params['latent_size_gnn']
            self.processor_nodes.append(generate_proc_MLP(lsgnn * 2))
            self.processor_edges.append(generate_proc_MLP(lsgnn * 3))

        self.output = MLP(params['latent_size_gnn'],
                          2,
                          params['latent_size_mlp'],
                          params['number_hidden_layers_mlp'],
                          False)

    def encode_nodes(self, nodes):
        enc_features = self.encoder_nodes(nodes.data['nfeatures'])
        return {'proc_node': enc_features}

    def encode_edges(self, edges):
        enc_features = self.encoder_edges(edges.data['efeatures'])
        return {'proc_edge': enc_features}

    def process_edges(self, edges, index):
        f1 = edges.data['proc_edge']
        f2 = edges.src['proc_node']
        f3 = edges.dst['proc_node']
        proc_edge = self.processor_edges[index](torch.cat((f1, f2, f3), 1))
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge': proc_edge}

    def process_nodes(self, nodes, index):
        f1 = nodes.data['proc_node']
        f2 = nodes.data['pe_sum']
        proc_node = self.processor_nodes[index](torch.cat((f1, f2), 1))
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node': proc_node}

    def decode_nodes(self, nodes):
        h = self.output(nodes.data['proc_node'])
        return {'pred_labels': h}

    def forward(self, g):
        g.apply_nodes(self.encode_nodes)
        g.apply_edges(self.encode_edges)
        
        for index in range(self.process_iters):
            def process_edges(edges):
                return self.process_edges(edges, index)
            def process_nodes(nodes):
                return self.process_nodes(nodes, index)
            # compute junction-branch interactions
            g.apply_edges(process_edges)
            g.update_all(fn.copy_e('proc_edge', 'm'), 
                         fn.sum('m', 'pe_sum'))
            g.apply_nodes(process_nodes)

        g.apply_nodes(self.decode_nodes)

        return g.ndata['pred_labels']