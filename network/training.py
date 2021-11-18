import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("core/")

import torch
import preprocessing as pp
from graphnet import GraphNet
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def main(model):
    dataset = pp.generate_dataset(model)
    num_examples = len(dataset)
    print(num_examples)
    num_train = int(num_examples)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=10, drop_last=False)

    infeat_nodes = 8
    infeat_edges = 5
    latent_size = 16
    out_size = 2
    process_loop = 1
    hl_mlp = 2
    model = GraphNet(infeat_nodes, infeat_edges, latent_size,
                     out_size, process_loop, hl_mlp)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # lr=0.01, weight_decay=0.0001)
    nepochs = 5000
    for epoch in range(nepochs):
        print('ep = ' + str(epoch))
        global_loss = 0
        count = 0
        for batched_graph in train_dataloader:
            pred = model(batched_graph, batched_graph.ndata['n_features'].float()).squeeze()
            weight = torch.ones(pred.shape)
            # mask out values corresponding to boundary conditions
            weight[batched_graph.ndata['inlet_mask'],1] = 0
            weight[batched_graph.ndata['outlet_mask'],0] = 0
            loss = weighted_mse_loss(pred, torch.reshape(batched_graph.ndata['n_labels'].float(),
                                     pred.shape), weight)
            global_loss = global_loss + loss.detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = count + 1

        print('\tloss = ' + str(global_loss / count))

    it = iter(train_dataloader)
    batch = next(it)
    batched_graph, labels = batch
    graph = dgl.unbatch(batched_graph)[0]
    timeg = float(graph.ndata['time'][0])
    nodes_degree = np.expand_dims(graph.ndata['features'][:,-1],1)



if __name__ == "__main__":
    main(sys.argv[1])
