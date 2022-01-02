import torch
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import LeakyReLU
import torch.nn.functional as F
import dgl.function as fn
from torch.nn import Dropout
import numpy as np

class MLP(Module):
    def __init__(self, in_feats, latent_space, out_feats, n_h_layers, normalize = True):
        bound_init = 1
        super().__init__()
        self.encoder_in = Linear(in_feats, latent_space, bias = True).float()
        # torch.nn.init.uniform_(self.encoder_in.weight, -bound_init, bound_init)
        self.encoder_out = Linear(latent_space, out_feats, bias = True).float()
        # torch.nn.init.uniform_(self.encoder_out.weight, -bound_init, bound_init)

        self.n_h_layers = n_h_layers
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_h_layers):
            self.hidden_layers.append(Linear(latent_space, latent_space, bias = True).float())
            # torch.nn.init.uniform_(self.hidden_layers[i].weight, -bound_init, bound_init)

        self.normalize = normalize
        if self.normalize: # Try GroupNorm
            self.norm = LayerNorm(out_feats).float()

    def forward(self, inp):
        enc_features = self.encoder_in(inp)
        enc_features = F.leaky_relu(enc_features) # try leaky relu (0.1) or elu

        for i in range(self.n_h_layers):
            enc_features = self.hidden_layers[i](enc_features)
            enc_features = F.leaky_relu(enc_features)

        enc_features = self.encoder_out(enc_features)

        if self.normalize:
            enc_features = self.norm(enc_features)

        return enc_features

class GraphNet(Module):
    def __init__(self, params):
        super(GraphNet, self).__init__()
        
        self.encode_inlet_edge = MLP(7, 16, 8, 1, True)
        self.encode_outlet_edge = MLP(7, 16, 8, 1, True)

        self.encoder_nodes = MLP(params['infeat_nodes'],
                                 params['latent_size_mlp'],
                                 params['latent_size_gnn'],
                                 params['hl_mlp'],
                                 params['normalize'])
        self.encoder_edges = MLP(params['infeat_edges'],
                                 params['latent_size_mlp'],
                                 params['latent_size_gnn'],
                                 params['hl_mlp'],
                                 params['normalize'])

        self.processor_edges = torch.nn.ModuleList()
        self.processor_nodes = torch.nn.ModuleList()
        self.process_iters = params['process_iterations']
        for i in range(self.process_iters):
            self.processor_edges.append(MLP(params['latent_size_gnn'] * 3,
                                        params['latent_size_mlp'],
                                        params['latent_size_gnn'],
                                        params['hl_mlp'],
                                        params['normalize']))
            self.processor_nodes.append(MLP(params['latent_size_gnn'] * 2,
                                        params['latent_size_mlp'],
                                        params['latent_size_gnn'],
                                        params['hl_mlp'],
                                        params['normalize']))

        self.output = MLP(params['latent_size_gnn'],
                          params['latent_size_mlp'],
                          params['out_size'],
                          params['hl_mlp'],
                          False)

        # self.dropout = Dropout(0.5)

    def encode_nodes(self, nodes):
        f = nodes.data['features_c']
        enc_features = self.encoder_nodes(f)
        return {'proc_node': enc_features}

    def encode_edges(self, edges):
        f = edges.data['e_features']
        enc_features = self.encoder_edges(f)
        return {'proc_edge': enc_features}

    def process_edges(self, edges, layer):
        f1 = edges.data['proc_edge']
        f2 = edges.src['proc_node']
        f3 = edges.dst['proc_node']
        proc_edge = self.processor_edges[layer](torch.cat((f1, f2, f3),dim=1))
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge' : proc_edge}

    def process_nodes(self, nodes, layer):
        f1 = nodes.data['proc_node']
        f2 = nodes.data['pe_sum']
        proc_node = self.processor_nodes[layer](torch.cat((f1, f2),dim=1))
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node' : proc_node}

    def decode(self, nodes):
        f = nodes.data['proc_node']
        # f = self.dropout(f)
        h = self.output(f)
        return {'h' : h}

    # def forward(self, g, in_feat):
    #     g.ndata['features_c'] = in_feat
    #     g.apply_nodes(self.encode_nodes)
    #     g.apply_edges(self.encode_edges)
    #     for i in range(self.process_iters):
    #         def pe(edges):
    #             return self.process_edges(edges, i)
    #         def pn(nodes):
    #             return self.process_nodes(nodes, i)
    #         g.apply_edges(pe)
    #         # aggregate new edge features in nodes
    #         g.update_all(fn.copy_e('proc_edge', 'm'), fn.sum('m', 'pe_sum'))
    #         g.apply_nodes(pn)
    #     g.apply_nodes(self.decode)
    #     return g.ndata['h']
    
    def forward(self, g, in_feat):
        g.nodes['inner'].data['features_c'] = in_feat
        g.update_all(fn.u_add_v('proc_node', 'proc_node', 'm'), fn.sum('m', 'pe_sum'))
        g.apply_nodes(self.encode_nodes, ntype='inner')
        for i in range(self.process_iters):
            def pn(nodes):
                return self.process_nodes(nodes, i)
            g.update_all(fn.u_add_v('proc_node', 'proc_node', 'm'), fn.sum('m', 'pe_sum'))
            g.apply_nodes(pn)
        g.apply_nodes(self.decode)
        return g.ndata['h']
