
import argparse
import multiprocessing as mp
import os
import warnings
from copy import deepcopy

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import torch
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import add_self_loops

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)

class MyDataset(InMemoryDataset):
    def __init__(self, root, A, links, labels, h, sample_ratio, max_nodes_per_hop, 
                 features, drug_sim,dis_sim, class_values, max_num=None,parallel=False):
        self.Arow = SparseRowIndexer(A[0])
        self.Acol = SparseColIndexer(A[0].tocsc())
        self.geneArow = SparseRowIndexer(A[1])
        self.drugArow = SparseRowIndexer(A[2])
        self.diseaseArow = SparseRowIndexer(A[3])
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        # self.u_features = u_features
        # self.v_features = v_features
        self.class_values = class_values
        self.parallel = parallel
        # self.rel_features = rel_features
        self.features = features
        self.max_num = max_num
        # self.gene_path = gene_path
        self.num_drugs = self.drugArow.shape[0]
        self.num_diseases = self.diseaseArow.shape[0]
        self.num_genes = self.drugArow.shape[1]
        self.num_nodes = self.num_drugs+self.num_diseases+self.num_genes
        self.drug_sim = drug_sim
        self.dis_sim = dis_sim

        self.mlb = LabelBinarizer()
        if max(self.class_values) > 1:
            self.mlb.fit(np.array(class_values))
        else:
            # self.mlb.fit(np.array([-1.0, 0.0, 1.0]))
            self.mlb.fit(np.array([1.0]))
        
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'data.pt'
        if self.max_num is not None:
            name = 'data_{}.pt'.format(self.max_num)
        return [name]

    def edge_features(self):
        if len(set(self.class_values)) == 2:
            return 3
        return len(set(self.class_values))

    def process(self):
        # Extract enclosing subgraphs and save to disk
        # tmp = onegraph(self.Arow, self.mlb, self.features, self.gene_path)
        tmp = onegraph([self.Arow,self.geneArow,self.drugArow,self.diseaseArow], self.mlb, self.features)
        data_list = construct_pyg_graph(None,*tmp,self.drug_sim,self.dis_sim,y=self.labels)
        data_list.links = torch.LongTensor(self.links)
        data_list.num_drugs = self.num_drugs
        data, slices = self.collate([data_list])
        torch.save((data, slices), self.processed_paths[0])
        del data_list


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels, h, sample_ratio, max_nodes_per_hop, 
                 features, class_values, gene_path,max_num=None):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A[0])
        self.Acol = SparseColIndexer(A[0].tocsc())
        self.geneArow = SparseRowIndexer(A[1])
        self.drugArow = SparseRowIndexer(A[2])
        self.diseaseArow = SparseRowIndexer(A[3])
        self.links = links
        self.labels = labels
        self.gene_path = gene_path
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.class_values = class_values
        self.features = features
        self.num_features_genecd = features[-1].shape[1]
        # self.num_features = features[-1].shape[1]
        self.num_features_cd = features[0].shape[1]+8
        self.num_edge_features_genecd = features[0].shape[1]
        self.num_edge_features_cd = 20
        self.mlb = LabelBinarizer()
        if max(self.class_values) > 1:
            self.mlb.fit(np.array(class_values))
        else:
            # self.mlb.fit(np.array([-1.0, 0.0, 1.0]))
            self.mlb.fit(np.array([1.0]))

        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def __len__(self):
        return len(self.links[0])

    def edge_features(self):
        if len(set(self.class_values)) == 2:
            return 3
        return len(set(self.class_values))

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]

        tmp_cd = subgraph_extraction_labeling(
            self.Arow, self.features, self.mlb
        )

        paths = get_genepath(self.drugArow,self.diseaseArow,self.gene_path,self.features,i,j)
        data_cd = construct_pyg_graph([i,j+self.Arow.shape[0]], paths, *tmp_cd, g_label)
        self.num_features_cd = data_cd.x.shape[1]
        self.num_edge_features_cd = data_cd.edge_attr.shape[1]
        return data_cd
        # return data_genecd

def onegraph(Arow,mlb,features):
    Arow,geneArow,drugArow,diseaseArow = Arow
    graph = Arow[:][:]
    u, v, r = ssp.find(graph)
    v += Arow.shape[0]
    if max(r) == 1:
        newr = [float(i) if i == 1 else -1 for i in r]
        attr = mlb.transform(newr).astype(dtype=np.int8)
    else:
        attr = mlb.transform(r).astype(dtype=np.int8) # attr 的shape
    if features == None:
        drug_features = None
        disease_features = None
        all_node_features = None
        node_features = None
    else:
        drug_features = np.array(features[0])
        disease_features = np.array(features[1])
        node_features = np.concatenate((drug_features, disease_features))
        rel_features = np.array(features[2])
        all_node_features = np.concatenate((features[3], drug_features, disease_features))
    # paths = gene_path.reshape((-1,gene_path.shape[-2],gene_path.shape[-1]))
    return u, v, r, node_features, attr,all_node_features,rel_features
    # return paths, u, v, r, node_features, attr

def subgraph_extraction_labeling(Arow, features, mlb=None):
    subgraph = Arow[:][:]
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1) 
    v += Arow.shape[0]
    r = r - 1 
    if max(r) == 1:
        newr = [float(i) if i == 1 else -1 for i in r]
        attr = mlb.transform(newr).astype(dtype=np.int8)
    else:
        attr = mlb.transform(r).astype(dtype=np.int8) 
    num_nodes = sum(Arow.shape)
    
    # get node features
    if features is not None:
        drug_features = np.array(features[0])
        disease_features = np.array(features[1])

    if drug_features is not None and disease_features is not None:
        node_features = np.concatenate((drug_features, disease_features))
    
    return u, v, r, node_features, attr

def retrive_path(gene,gene_path,features):
    paths = gene_path[list(gene)]
    paths = paths.reshape((-1,paths.shape[-2],paths.shape[-1]))
    rand_idx = list(range(paths.shape[0]))
    np.random.shuffle(rand_idx)
    paths = paths[rand_idx]
    addcol = np.full((len(paths),1,200),features)
    paths = np.concatenate((addcol,paths),axis=1)
    if paths.shape[0] > 1000:
        return paths[:1000]
    return paths

def get_genepath(drugArow,diseaseArow,gene_path,features,drug,disease):
    drug_features = features[0][drug]
    disease_features = features[1][disease]
    drug_gene = neighbors([drug],drugArow)
    disease_gene = neighbors([disease],diseaseArow)
    drug_path = retrive_path(drug_gene,gene_path,drug_features)
    disease_path = retrive_path(disease_gene,gene_path,disease_features)
    return (drug_path,disease_path)

def construct_pyg_graph(ind, u, v, r, node_features, attr, all_node_features,rel_features,drug_sim,dis_sim, y):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    # edge_index = add_self_loops(edge_index)[0]
    edge_type = torch.cat([r, r])
    edge_index, edge_type = add_self_loops(edge_index,edge_type,fill_value=1)
    attr = torch.FloatTensor(attr)
    edge_attr = torch.cat([attr, attr,attr[:node_features.shape[0]]], dim=0)
    x = torch.FloatTensor(node_features) # node_features = np.concatenate((drug_features, disease_features))
    y = torch.Tensor(y)
    x_onehot = torch.eye(len(x))
    # all_onehot = torch.eye(len(all_node_features))
    all_onehot = None
    all_node_features = torch.FloatTensor(all_node_features)
    rel_features = torch.FloatTensor(rel_features)
    drug_sim = torch.FloatTensor(drug_sim)
    dis_sim = torch.FloatTensor(dis_sim)
    if ind!=None:
        idx = torch.zeros(len(x))
        idx[ind[0]] = 1
        idx[ind[1]] = 2
        # paths = (torch.FloatTensor(paths[0]),torch.FloatTensor(paths[1]))
    else:
        idx = None
        data = Data(x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, \
                idx=idx, all_node_features=all_node_features,rel_features=rel_features,y=y, drug_sim = drug_sim,dis_sim = dis_sim)
 
    return data


def onehot_encoding(x, allowable_set):
    return [x == s for s in allowable_set]


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def path(fringe,A):
    if not fringe:
        return set([])
    else:
        nei = A[list(fringe)]

def node_score(fringe,A):

    num_neis = []
    for i in fringe:
        i = int(i)
        num_nei = len(A[[i]].indices)
        num_neis.append(num_nei)
    return num_neis

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    # transform r back to rating label
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g
