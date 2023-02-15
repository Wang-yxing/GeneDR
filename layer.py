
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

class LSTM_aggr(MessagePassing):
    def __init__(self, feature_length, walk_len, hidden_size, dropout,per_path):
        super(LSTM_aggr,self).__init__()
        self.per_path = per_path
        self.feature_length, self.hidden_size, self.walk_len = feature_length, hidden_size, walk_len
        # self.fc0 = torch.nn.Linear(feature_length, hidden_size)
        # self.nets = torch.nn.ModuleList(
        #     [torch.nn.Linear(hidden_size, hidden_size) for i in range(self.walk_len)])
        self.dropout = dropout
        self.LSTM = torch.nn.LSTM(feature_length, hidden_size)
        # self.attw = torch.nn.Linear(2 * hidden_size, 1)
        self.attw = torch.nn.Linear(hidden_size+200, 1)
        # self.attw = torch.nn.Parameter(torch.full((self.per_path,866,1),1/self.per_path))
        self.Lrelu = torch.nn.LeakyReLU()

    def forward(self, thepath, X, onegraph=True):
        feature_dim = thepath[0][0].shape[-1]
        # per_path = 10
        output = []
        
        if onegraph:
            paths = thepath.transpose(0,1).reshape(-1,thepath.shape[-1])
        else:
            indxx = []
            allpath = []
            for idx,paths in enumerate(thepath):
                indxx.extend([idx*2]*paths[0].shape[0])
                indxx.extend([idx*2+1]*paths[1].shape[0])
                paths = torch.cat((paths[0],paths[1]))
                allpath.append(paths)
            paths = torch.cat(allpath,dim=0).transpose(0,1)

        path = torch.flip(paths,dims=[1]).transpose(0,1)
        path = X[path]
        path = F.dropout(path, p=0.7, training=self.training)
        path, (h_n, c_n) = self.LSTM(path)
        # h_n = torch.cat([path[0],path[1],path[2],path[3]],1).unsqueeze(0)
        # h_n = torch.tanh(h_n)
        # h_n = h_n.transpose(0, 1).view(num_w,-1,self.hidden_size)
        if onegraph:
            # h_n = h_n.squeeze().reshape(-1,self.per_path,self.hidden_size)
            h_n = h_n.transpose(0, 1).view(self.per_path, -1, self.hidden_size)
            origin = X[thepath.transpose(0,1)[:,:,0]]
            cat_res = torch.cat((h_n,origin),dim=-1)
            # att_score = F.softmax(self.Lrelu(self.attw(cat_res)))
            # # h_n = att_score * h_n
            # h_n = att_score * h_n
            output = torch.mean(cat_res, dim=0)
        else:
            for idx in range(len(thepath)):
                h_n_drug = h_n.squeeze()[np.array(indxx)==idx*2]
                h_n_disease = h_n.squeeze()[np.array(indxx)==idx*2+1]
                att_score = F.softmax(self.Lrelu(self.attw(h_n_drug)))  # num_w, split, 1
                h_n_drug = att_score * h_n_drug
                drug_out = torch.mean(h_n_drug, dim=0)
                drug_out = torch.cat((paths[0,0,:], drug_out))

                att_score = F.softmax(self.Lrelu(self.attw(h_n_disease)))  # num_w, split, 1
                h_n_disease = att_score * h_n_disease
                disease_out = torch.mean(h_n_disease, dim=0)
                disease_out = torch.cat((paths[0,-1,:], disease_out))
                output.append([drug_out,disease_out])
        return output
