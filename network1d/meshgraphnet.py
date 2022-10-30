import os
import sys
sys.path.append(os.getcwd())
import torch as th
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
import graph1d.generate_normalized_graphs as nz

class MLP(Module):
    """
    Multi-layer perceptron.
    
    Attributes:
        input: Linear pytorch module
        output: Linear pytorch module
        n_h_layers (int): number of hidden layers
        hidden_layers: list of Linear modules
        normalize (bool): specifies if last LayerNorm should be applied after 
                          last layer
        norm: LayerNorm pytorch module 
  
    """
    def __init__(self, in_feats, out_feats, latent_space, n_h_layers, 
                normalize = True):
        """
        Init MLP.
        
        Initialize MLP.

        Arguments:
            in_feats (int): number of input features
            out_feats (int): number of output features
            latent_space (int): size of the latent space
            n_h_layers (int): number of hidden layers
            normalize (bool): specifies whether normalization should be applied
                              in last layer. Default -> true 
    
        """
        super().__init__()
        self.input = Linear(in_feats,latent_space,bias = True).float()
        self.output = Linear(latent_space, out_feats, bias = True).float()
        self.n_h_layers = n_h_layers
        self.hidden_layers = th.nn.ModuleList()
        for i in range(self.n_h_layers):
            self.hidden_layers.append(Linear(latent_space, 
                                             latent_space, 
                                             bias = True).float())

        self.normalize = normalize
        if self.normalize:
            self.norm = LayerNorm(out_feats).float()

    def forward(self, inp):
        """
        Forward step

        Arguments:
            inp: input tensor

        Returns:
            result of forward step

        """
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
    """
    MeshGraphNet

    This class computes pressure and flowrate updates given the previous system
    state.
    
    Attributes:
        params: dictionary of hyperparameters
        encoder_nodes: MLP encoding graph nodes
        encoder_edges: MLP encoding graph edges
        processor_nodes: MLP processing nodes in network layers
        processor_edges: MLP processing edges in network layers
        process_iters: number of iterations within the network
        output: MLP deconing graph nodes
  
    """
    def __init__(self, params):
        """
        Init MeshGraphNet

        Arguments:
            params: dictionary of hyperparameters

        """
        super(MeshGraphNet, self).__init__()

        self.params = params

        self.encoder_nodes = MLP(params['infeat_nodes'], 
                                 params['latent_size_gnn'],
                                 params['latent_size_mlp'],
                                 params['number_hidden_layers_mlp'])
        self.encoder_edges = MLP(params['infeat_edges'], 
                                 params['latent_size_gnn'],
                                 params['latent_size_mlp'],
                                 params['number_hidden_layers_mlp'])


        self.processor_nodes = th.nn.ModuleList()
        self.processor_edges = th.nn.ModuleList()
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
                          params['out_size'],
                          params['latent_size_mlp'],
                          params['number_hidden_layers_mlp'],
                          False)

        if params['bc_type'] == 'physiological':
            self.inlet_encoder = MLP(4,
                                     params['latent_size_gnn'],
                                     params['latent_size_mlp'],
                                     params['number_hidden_layers_mlp'])
            self.outlet_RCR_encoder = MLP(6,
                                          params['latent_size_gnn'],
                                          params['latent_size_mlp'],
                                          params['number_hidden_layers_mlp'])

    def encode_nodes(self, g):
        """
        Encode graph nodes

        Arguments:
            g: the graph

        """
        if self.params['bc_type'] == 'physiological':
            inmask = g.ndata['inlet_mask']
            outmask = g.ndata['outlet_mask']
            mask = (th.ones(inmask.shape) - inmask - outmask).bool()
            enc_features = self.encoder_nodes(g.ndata['nfeatures'][mask,:])
            g.ndata['proc_node'][mask,:] = enc_features
        else:
            enc_features = self.encoder_nodes(g.ndata['nfeatures'].clone())
            g.ndata['proc_node'] = enc_features

    def encode_edges(self, edges):
        """
        Encode graph edges

        Arguments:
            edges: graph edges

        Returns:
            dictionary (key: 'proc_edge', value: encoded features)

        """
        enc_features = self.encoder_edges(edges.data['efeatures'])
        return {'proc_edge': enc_features}

    def process_edges(self, edges, index):
        """
        Process graph edges

        Arguments:
            edges: graph edges
            index: iteration index

        Returns:
            dictionary (key: 'proc_edge', value: processed features)

        """
        f1 = edges.data['proc_edge']
        f2 = edges.src['proc_node']
        f3 = edges.dst['proc_node']
        proc_edge = self.processor_edges[index](th.cat((f1, f2, f3), 1))
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge': proc_edge}

    def process_nodes(self, nodes, index):
        """
        Process graph nodes

        Arguments:
            nodes: graph nodes
            index: iteration index

        Returns:
            dictionary (key: 'proc_node', value: processed features)

        """
        f1 = nodes.data['proc_node']
        f2 = nodes.data['pe_sum']
        proc_node = self.processor_nodes[index](th.cat((f1, f2), 1))
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node': proc_node}

    def decode_nodes(self, nodes):
        """
        Decode graph nodes

        Arguments:
            nodes: graph nodes

        Returns:
            dictionary (key: 'pred_labels', value: decoded features)

        """
        h = self.output(nodes.data['proc_node'])
        return {'pred_labels': h}

    def continuity_loss(self, g, flowrate, take_mean = True):
        """
        Compute contiuity loss

        Continuity loss as the mass loss occurring  at junctions.

        Arguments:
            g: graph
            flowrate: tensor containing nodal values of flowrate
            take_mean: if True, take mean of junction losses. If 
                       False, take sum. Default -> True.
        Returns: 
            sum of mass loss occurring at branches and at junctions

        """
        g.ndata['next_flowrate'] = flowrate.clone()

        # we zero-out inlet and outlet flowrate (otherwise they would send
        # their flowrate to branch and junction nodes)
        g.ndata['next_flowrate'][g.ndata['inlet_mask'].bool()] = 0
        g.ndata['next_flowrate'][g.ndata['outlet_mask'].bool()] = 0

        # # we send flowrate through branches, compute the mean
        # # of neighboring nodes, and compute the diff with our estimate
        # g.update_all(fn.copy_u('next_flowrate', 'm'), 
        #              fn.sum('m', 'sum_flowrate'))
        # # branch nodes have only two neighbors
        # diff = th.abs(2 * g.ndata['next_flowrate'] - g.ndata['sum_flowrate'])
        # diff = diff * g.ndata['continuity_mask']
        # if take_mean:
        #     branch_continuity = th.mean(diff)
        # else:
        #     branch_continuity = th.sum(diff)

        # we keep flowrate at inlet and outlets of junctions
        g.ndata['flow_junction'] = g.ndata['next_flowrate'] * \
                                   g.ndata['jun_mask']

        g.update_all(fn.copy_u('flow_junction', 'm'), 
                     fn.sum('m', 'sum_flowrate'))

        # we use the inlet to compute the difference
        diff = th.abs(g.ndata['sum_flowrate'] - g.ndata['next_flowrate'])
        diff = diff * g.ndata['jun_inlet_mask']

        if take_mean:
            junction_continuity = th.sum(diff) / \
                                th.sum(g.ndata['jun_inlet_mask'])
        else:
            junction_continuity = th.sum(diff)

        return junction_continuity

    def encode_inlet(self, g):
        """
        Encode inlet using next_flowrate value

        Arguments:
            g: the graph
        """
        in_mask = g.ndata['inlet_mask']
        curvalues = g.ndata['nfeatures'][in_mask.bool(),0:3]
        nf = th.unsqueeze(g.ndata['next_flowrate'][in_mask.bool()],1)
        h = self.inlet_encoder(th.cat((curvalues, nf), 1))
        g.ndata['proc_node'][in_mask.bool(),0:] = h

    def encode_outlets(self, g):
        """
        Encode outlets using rcr values

        Arguments:
            g: the graph
        """
        out_mask = g.ndata['outlet_mask']
        curvalues = g.ndata['nfeatures'][out_mask.bool(),0:3]
        r1 = th.reshape(g.ndata['resistance1'][out_mask.bool()],(-1,1))
        c = th.reshape(g.ndata['capacitance'][out_mask.bool()],(-1,1))
        r2 = th.reshape(g.ndata['resistance2'][out_mask.bool()],(-1,1))
        h = self.outlet_RCR_encoder(th.cat((curvalues, r1, c, r2), 1))
        g.ndata['proc_node'][out_mask.bool(),:] = h

    def forward(self, g):
        """
        Forward step

        Arguments:
            g: the graph

        Returns:
            n x 2 tensor (n number of nodes in the graph) containing the update
                for pressure (first column) and the update for the flowrate 
                (second column)

        """
        if 'proc_node' not in g.ndata:
            nnodes = g.ndata['nfeatures'].shape[0]
            size = self.params['latent_size_gnn']
            g.ndata['proc_node'] = th.zeros((nnodes, size))
            
        if self.params['bc_type'] == 'physiological':
            self.encode_inlet(g)
            self.encode_outlets(g)

        # g.apply_nodes(self.encode_nodes)
        self.encode_nodes(g)
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