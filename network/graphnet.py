import torch
from torch.nn.modules.module import Module
from dgl.nn import ChebConv
from dgl.nn import GraphConv
from dgl.nn import GATConv
from dgl.nn import RelGraphConv
from torch.nn import LayerNorm
from torch.nn import Linear
import torch.nn.functional as F
import dgl.function as fn

class MLP(Module):
    def __init__(self, in_feats, latent, n_h_layers, normalize = True):
        super().__init__()
        self.encoder_in = Linear(in_feats, latent).float()
        self.encoder_out = Linear(latent, latent).float()

        self.n_h_layers = n_h_layers
        self.hidden_layers = []
        for i in range(n_h_layers):
            self.hidden_layers.append(Linear(latent, latent).float())

        self.normalize = normalize
        if self.normalize:
            self.norm = LayerNorm(latent).float()

    def forward(self, inp):
        enc_features = self.encoder_in(inp)
        enc_features = F.relu(enc_features)

        for i in range(self.n_h_layers):
            enc_features = self.hidden_layers[i](enc_features)
            enc_features = F.relu(enc_features)

        enc_features = self.encoder_out(enc_features)

        if self.normalize:
            enc_features = self.norm(enc_features)

        return enc_features

class GraphNet(Module):
    def __init__(self, in_feats_nodes, in_feats_edges, latent, h_feats, L, hidden_layers):
        super(GraphNet, self).__init__()

        normalize_inner = True

        self.encoder_nodes = MLP(in_feats_nodes, latent, hidden_layers, normalize_inner)
        self.encoder_edges = MLP(in_feats_edges, latent, hidden_layers, normalize_inner)

        self.processor_edges = []
        self.processor_nodes = []
        for i in range(L):
            self.processor_edges.append(MLP(latent * 3, latent, hidden_layers, normalize_inner))
            self.processor_nodes.append(MLP(latent * 2, latent, hidden_layers, normalize_inner))

        self.L = L

        self.output = MLP(latent, h_feats, hidden_layers, False)

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
        h = self.output(f)
        return {'h' : h}

    def forward(self, g, in_feat):
        g.ndata['features_c'] = in_feat
        g.apply_nodes(self.encode_nodes)
        g.apply_edges(self.encode_edges)
        for i in range(self.L):
            def pe(edges):
                return self.process_edges(edges, i)
            def pn(nodes):
                return self.process_nodes(nodes, i)
            g.apply_edges(pe)
            # aggregate new edge features in nodes
            g.update_all(fn.copy_e('proc_edge', 'm'), fn.sum('m', 'pe_sum'))
            g.apply_nodes(pn)
        g.apply_nodes(self.decode)
        return g.ndata['h']
