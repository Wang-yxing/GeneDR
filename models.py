
import time
import torch
import torch.nn.functional as F
from dataset import *
from layer import LSTM_aggr
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from utils import dropout_adj




class GeneDR(torch.nn.Module):
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 32], 
                 num_relations=5, regression=False, adj_dropout=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1,per_path=100):
        super(GeneDR, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by # 这个是啥
        self.num_relations = num_relations
        self.num_nodes = dataset.num_drugs+dataset.num_diseases
        self.num_genes = dataset.num_genes
        self.per_path = per_path
        self.x_lin = Linear(dataset.num_features,128)
        self.x_lin2 = Linear(dataset.num_features,128)
        self.convs_cd = torch.nn.ModuleList()
        self.convs_cd.append(gconv(latent_dim[0]*4, latent_dim[0]*2))
        for i in range(0, len(latent_dim)-1):
            self.convs_cd.append(gconv(latent_dim[i]*4, latent_dim[i+1]*2))
        self.convs_lstm = torch.nn.ModuleList()
        self.convs_lstm.append(LSTM_aggr(dataset.num_features,7,256,0.2,self.per_path))
        for i in range(0, len(latent_dim)):
            self.convs_lstm.append(LSTM_aggr(256,7,256,0.2,self.per_path))
        self.lin1 = Linear(128, 2 * 128)
        self.lin2 = Linear(256 ,1)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(4*(sum(latent_dim)), 2 * 128)
        self.nn_cd = torch.nn.Linear(dataset.num_edge_features,128)
        self.att_score = torch.nn.Parameter(torch.tensor([1/3,1/3,1/3]))
        self.relu = torch.nn.PReLU()
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))
        
    def forward(self, data,onegraph=True):
        start = time.time()

        x_cd, edge_index_cd, edge_type_cd, edge_attr_cd,drug_sim,dis_sim = data.x, data.edge_index, data.edge_type, data.edge_attr,data.drug_sim,data.dis_sim
        paths = data.paths
        # all_onehot = data.all_onehot
        if self.adj_dropout > 0:
            edge_index_cd, edge_type_cd, edge_attr_cd = dropout_adj(
                edge_index_cd, edge_type_cd, edge_attr_cd, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x_cd), 
                training=self.training
            )
        edge_attr_cd = self.nn_cd(edge_attr_cd)
        x_cd = self.x_lin2(x_cd)
        concat_states_cd = []
        concat_states_lstm = []

        x_all = torch.cat([data.all_node_features,data.rel_features])

        for conv,lstm in zip(self.convs_cd,self.convs_lstm):
            h_n = self.relu(lstm(paths,x_all))
            x = self.l2_norm(conv(h_n, edge_index_cd))
            concat_states_cd.append(x)
            # concat_states_cd.append(x)
            concat_states_lstm.append(h_n)
            x_all[self.num_genes:self.num_nodes+self.num_genes] = x
        concat_states_cd = torch.cat(concat_states_cd, 1)
        concat_states_lstm = torch.cat(concat_states_lstm, 1)

        if onegraph:
            x_cd = concat_states_cd
            drug_users = x_cd[:data.num_drugs]
            disease_items = x_cd[data.num_drugs:]
            x_cd = torch.cat([drug_users[data.links[0]],disease_items[data.links[1]]],1)

        x_cd = self.relu(self.lin1(x_cd))
        x_cd = F.dropout(x_cd, p=0.7, training=self.training)

        x_cd = self.lin2(x_cd)
        h_n=0
        # x_all=0
        return x_cd[:, 0],(h_n,concat_states_cd),(x_all)


