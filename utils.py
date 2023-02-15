
import json
import os
import subprocess
from turtle import distance
from unicodedata import name

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_sparse import coalesce


def maybe_num_nodes(index,num_nodes= None):
    return int(index.max()) + 1 if num_nodes is None else num_nodes

def filter_adj(row, col, edge_type, edge_attr, mask):
    return row[mask], col[mask], None if edge_type is None else edge_type[mask],None if edge_attr is None else edge_attr[mask]

def dropout_adj(edge_index, edge_type=None, edge_attr=None, p=0.5, force_undirected=False,
                num_nodes=None, training=True):
    r"""Randomly drops edges from the adjacency matrix
    :obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
    """

    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training:
        return edge_index, edge_type, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_type, edge_attr = filter_adj(row, col, edge_type, edge_attr, row < col)

    mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)

    row, col, edge_type, edge_attr = filter_adj(row, col, edge_type, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        if edge_type is not None:
            edge_type = torch.cat([edge_type, edge_type], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        edge_index, edge_type = coalesce(edge_index, edge_type, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_type, edge_attr


def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data
    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array([id_dict[x] for x in data])
    n = len(uniq)

    return data, id_dict, n

def map_data_forDGD(data,id_dict):
    """
    Map data to proper indices in case they are not in a continues [0, N) range
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data
    """
    uniq = list(set(data))
    idx = len(id_dict)
    dict_new = {}
    for i in sorted(uniq):
        if i in id_dict.keys():
            dict_new[i] = id_dict[i]
        else:
            dict_new[i]=idx
            idx+=1
    data = np.array([dict_new[x] for x in data])
    n = len(uniq)

    return data, dict_new, n


def name_id_convert(namelist):
    id2name = {}
    name2id = {}
    for id,i in enumerate(namelist):
        id2name[id] = i
        name2id[i] = id
    num = len(namelist)
    return id2name,name2id,num



def load_data_from_database(dataset='DATASET2', mode='transductive', testing=False, rating_map=None, post_rating_map=None, ratio=1.0,use_features=False,with_addds=False,posneg_ratio=1.0,per_path=None,walk_len=None,cross_fold=0.9):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """
    print("loading DD dataset ...")
    drug_disease_file = os.path.join('data/',dataset,'drug_disease_association.csv')
    if dataset == 'DATASET1':
        data_dr_di = pd.read_csv(
            drug_disease_file, usecols=['drug_id','disease_id']
        )
        data_dr_di.insert(loc=1,column='relation',value='dr-di')
        data_dr_di.columns=['sourceid','relation','targetid']
        lagcn_drug = pd.read_csv(
            'data/DATASET1/drug.csv', usecols=['drug_id','drugbank_id']
        )
        mesh2drugbank = {row['drug_id']:row['drugbank_id'] for _,row in lagcn_drug.iterrows()}
        
        for _,row in data_dr_di.iterrows():
            row['sourceid'] = mesh2drugbank[row['sourceid']]   
        
        data_array_dr_di = data_dr_di.values.tolist()
        data_array_dr_di = np.array(data_array_dr_di)         
        np.random.seed(10)
        np.random.shuffle(data_array_dr_di)

        u_nodes_ratings = data_array_dr_di[:,0]
        v_nodes_ratings = data_array_dr_di[:,2]
        ratings = data_array_dr_di[:,1]

        u_nodes, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes, v_dict, num_items = map_data(v_nodes_ratings) 
    
    if dataset == 'DATASET2':
        data_dr_di = pd.read_csv(
            drug_disease_file, usecols=['Drug','Disease']
        )
        data_dr_di.insert(loc=1,column='relation',value='dr-di')
        data_dr_di.columns=['sourceid','relation','targetid']
        data_array_dr_di = data_dr_di.values.tolist()
        data_array_dr_di = np.array(data_array_dr_di)         
        np.random.seed(10)
        np.random.shuffle(data_array_dr_di)

        data_dr_di_id = pd.read_csv(
            os.path.join('data/',dataset,'ID_drug_disease_association.csv'), usecols=['drug_id','disease_id']
        )
        data_dr_di_id.insert(loc=1,column='relation',value='dr-di')
        data_dr_di_id.columns=['sourceid','relation','targetid']

        u_file = pd.read_csv(
            'data/DATASET2/omics/drug.csv', 
        )
        u_dict = {i['Drug']:i['ID'] for _,i in u_file.iterrows()}
        num_users = len(u_dict)
        u_nodes = data_array_dr_di[:,0].astype(int)

        v_file = pd.read_csv(
            'data/DATASET2/omics/disease.csv', 
        )
        v_dict = {i['Disease']:i['ID'] for _,i in v_file.iterrows()}
        num_items = len(v_dict)
        v_nodes = data_array_dr_di[:,2].astype(int)
        ratings = data_array_dr_di[:,1]



    if dataset == 'F-dataset':
        data_dr_di = pd.read_csv(
            drug_disease_file, usecols=['drug','disease'],dtype=int
        )
        data_dr_di.insert(loc=1,column='relation',value='dr-di')
        data_dr_di.columns=['sourceid','relation','targetid']

        data_array_dr_di = data_dr_di.values.tolist()
        data_array_dr_di = np.array(data_array_dr_di)

        u_file = pd.read_csv(
            'data/F-dataset/drugName.csv', 
        )
        u_dict = {i['drugid']:i['idx'] for _,i in u_file.iterrows()}
        num_users = len(u_dict)
        u_nodes = data_array_dr_di[:,0].astype(int)

        v_file = pd.read_csv(
            'data/F-dataset/diseaseName.csv', 
        )
        v_dict = {i['diseaseid']:i['idx'] for _,i in v_file.iterrows()}
        num_items = len(v_dict)
        v_nodes = data_array_dr_di[:,2].astype(int)
        ratings = data_array_dr_di[:,1]

    # np.random.seed(10)
    # np.random.shuffle(data_array_dr_di)

    
    # u_nodes_ratings = data_array_dr_di[:,0]
    # v_nodes_ratings = data_array_dr_di[:,2]
    # ratings = data_array_dr_di[:,1]

    # u_nodes, u_dict, num_users = map_data(u_nodes_ratings)
    # v_nodes, v_dict, num_items = map_data(v_nodes_ratings) 

    neutral_rating = -1

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}
    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings]) # 这里直接就是che_dis.csv
    
    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])
    
    pairs_nonzero = [[u, v] for u, v in zip(u_nodes, v_nodes)]
    pairs_nonzero = np.array(pairs_nonzero)
    pairs_zero = np.array(np.where(labels==-1)).T

    labels = labels.reshape([-1])

    num_train = int(data_array_dr_di.shape[0]*cross_fold)
    num_test = len(data_array_dr_di)-num_train
    num_val = int(np.ceil(num_train * (1-cross_fold)))
    num_train = num_train - num_val

    np.random.shuffle(pairs_zero)
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])
    idx_zero = np.array([u * num_items + v for u, v in pairs_zero])

    rand_idx = list(range(len(idx_nonzero)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero = idx_nonzero[rand_idx]
    pairs_nonzero = pairs_nonzero[rand_idx]

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])
    posneg_ratio = float(pairs_zero.shape[0]/pairs_nonzero.shape[0])
    num_train_neg, num_test_neg, num_val_neg = int(posneg_ratio*num_train), int(posneg_ratio*num_test) ,int(posneg_ratio*num_val)

    val_idx = np.concatenate([idx_nonzero[0:num_val],idx_zero[0:num_val_neg]],axis=0)
    train_idx =  np.concatenate([idx_nonzero[num_val:num_train + num_val],idx_zero[num_val_neg:num_train_neg + num_val_neg]],axis=0)
    test_idx = np.concatenate([idx_nonzero[num_train + num_val:num_train + num_val + num_test],idx_zero[num_train_neg + num_val_neg:num_train_neg + num_val_neg + num_test_neg]],axis=0)

    val_pairs_idx = np.concatenate([pairs_nonzero[0:num_val],pairs_zero[0:num_val_neg]],axis=0)
    train_pairs_idx = np.concatenate([pairs_nonzero[num_val:num_train + num_val],pairs_zero[num_val_neg:num_train_neg + num_val_neg]],axis=0)
    test_pairs_idx = np.concatenate([pairs_nonzero[num_train + num_val:num_train + num_val + num_test],pairs_zero[num_train_neg + num_val_neg:num_train_neg + num_val_neg + num_test_neg]],axis=0)

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = np.concatenate([np.ones(num_train),np.zeros(num_train_neg)],axis=0)
    val_labels = np.concatenate([np.ones(num_val),np.zeros(num_val_neg)],axis=0)
    test_labels = np.concatenate([np.ones(num_test),np.zeros(num_test_neg)],axis=0)

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        train_idx = np.hstack([train_idx,val_idx])

    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32)+1
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    if with_addds:
        print("loading DGD dataset ...")
        data_DGD = pd.read_csv(
                # 'data/DATASET1/8k.csv',usecols=['sourceid','targetid','relation']
                os.path.join('data/',dataset,'8k.csv'),dtype=str
            )
        ##################
        data_DGD = process_pharmkg(data_DGD)
        ##################
        # data_DGD = data_DGD[
        #     ((data_DGD['sourcetype'].isin(['chemical','disease','gene']))&(data_DGD['targettype']=='gene'))|\
        #     ((data_DGD['targettype'].isin(['chemical','disease','gene']))&(data_DGD['sourcetype']=='gene'))
        #     ]

        data_DGD_cg = data_DGD[
            ((data_DGD['sourcetype'].isin(['chemical']))&(data_DGD['targettype']=='gene'))|\
            ((data_DGD['targettype'].isin(['chemical']))&(data_DGD['sourcetype']=='gene'))
            ]
        data_DGD_dg = data_DGD[
            ((data_DGD['sourcetype'].isin(['disease']))&(data_DGD['targettype']=='gene'))|\
            ((data_DGD['targettype'].isin(['disease']))&(data_DGD['sourcetype']=='gene'))
            ]
        data_DGD_gg = data_DGD[
            ((data_DGD['sourcetype'].isin(['gene']))&(data_DGD['targettype']=='gene'))
            ]

        data_DGD_dd = data_DGD[
            ((data_DGD['sourcetype'].isin(['disease']))&(data_DGD['targettype']=='disease'))
            ]
        data_DGD_cc = data_DGD[
            ((data_DGD['sourcetype'].isin(['chemical']))&(data_DGD['targettype']=='chemical'))
            ]
        data_DGD_cd = data_DGD[
            ((data_DGD['sourcetype'].isin(['chemical']))&(data_DGD['targettype']=='disease'))|\
            ((data_DGD['targettype'].isin(['chemical']))&(data_DGD['sourcetype']=='disease'))
            ]

        # data_DGD_cg = data_DGD_cg[['sourceid','relation','targetid']]
        data_DGD_cg = data_DGD_cg[data_DGD_cg['targetid'].notnull()&data_DGD_cg['sourceid'].notnull()]
        # data_DGD_dg = data_DGD_dg[['sourceid','relation','targetid']]
        data_DGD_dg = data_DGD_dg[data_DGD_dg['targetid'].notnull()&data_DGD_dg['sourceid'].notnull()]
        # data_DGD_gg = data_DGD_gg[['sourceid','relation','targetid']]
        data_DGD_gg = data_DGD_gg[data_DGD_gg['targetid'].notnull()&data_DGD_gg['sourceid'].notnull()]
        data_DGD_cg = priority(data_DGD_cg,'cg')
        data_DGD_dg = priority(data_DGD_dg,'dg')
        data_DGD_gg = priority(data_DGD_gg,'gg')
        
        data_DGD_cd = data_DGD_cd[data_DGD_cd['targetid'].notnull()&data_DGD_cd['sourceid'].notnull()]
        data_DGD_dd = data_DGD_dd[data_DGD_dd['targetid'].notnull()&data_DGD_dd['sourceid'].notnull()]
        data_DGD_cc = data_DGD_cc[data_DGD_cc['targetid'].notnull()&data_DGD_cc['sourceid'].notnull()]

        data_DGD = pd.concat([data_DGD_cg,data_DGD_dg,data_DGD_gg])
        data_toranse = pd.concat([data_DGD,data_DGD_dd,data_DGD_cc,data_DGD_cd])[['sourceid','relation','targetid']]
        data_toranse = data_toranse.values.tolist()
        data_toranse = np.array(data_toranse)

        dis_gene = pd.read_csv(
            os.path.join('data/',dataset,'dis_gene.csv'),header=None,usecols=[0,2]
        )
        dis_gene.insert(loc=1,column='relation',value='ML')
        dis_gene.columns=['sourceid','relation','targetid']
        che_gene = pd.read_csv(
            os.path.join('data/',dataset,'drug_gene.csv'),header=None,usecols=[1,2,4]
        )
        # che_gene.insert(loc=1,column='relation',value='ctd')
        che_gene.columns=['sourceid','targetid','relation']
        che_gene = CTDintegrate(che_gene)[['sourceid','relation','targetid']]
                
        temp = data_DGD[data_DGD['sourcetype']=='gene']['sourceid'].drop_duplicates()
        temp = temp[temp.notnull()]
        temp2 = data_DGD[data_DGD['targettype']=='gene']['targetid'].drop_duplicates()
        temp2 = temp2[temp2.notnull()]
        genes = list(temp)+list(temp2)
        genes = list(set(genes))
        gene_dict = {int(i):idx for idx,i in enumerate(genes)}
        drug_dict = u_dict
        disease_dict = v_dict
        num_gene = len(gene_dict)
        num_drug = len(drug_dict)
        num_disease = len(disease_dict)
        addgenes1 = pd.concat([dis_gene['sourceid'],che_gene['targetid']]).drop_duplicates()
        addgenes1 = list(addgenes1)
        idx = len(gene_dict)
        for i in addgenes1:
            if i not in gene_dict.keys():
                gene_dict[i] = idx
                idx += 1
        num_gene = len(gene_dict)

        data_DGD = data_DGD[['sourceid','relation','targetid']]
        data_array_DGD = data_DGD.values.tolist()
        data_array_DGD = np.array(data_array_DGD)

        data_array_all = np.concatenate([data_array_DGD,che_gene,dis_gene],axis=0)
        data_toranse = np.concatenate([data_toranse,che_gene,dis_gene],axis=0)
        data_toranse = np.array(data_toranse,dtype=str)
        node = np.unique(np.concatenate([data_toranse[:,0],data_toranse[:,2]],axis=0))
        rel = np.unique(data_toranse[:,1])
        # totranse(node,rel,data_toranse)


        u_nodes_ratings_p =  np.array(data_array_all[:,0],dtype=str)
        v_nodes_ratings_p =  np.array(data_array_all[:,2],dtype=str)
        ratings_p = data_array_all[:,1]

        # u_nodes_p, u_dict_p, num_users_p = map_data_forDGD(u_nodes_ratings_p,u_dict)
        # v_nodes_p, v_dict_p, num_items_p = map_data_forDGD(v_nodes_ratings_p,v_dict)
        


        rating_dict_p = {r: i for i, r in enumerate(np.sort(np.unique(ratings_p)).tolist())}
        # rating_dict.update(rating_dict_p)
        neutral_rating = -1
        # labels_p = np.full((num_users_p, num_items_p), neutral_rating, dtype=np.int32)
        # labels_p = np.array(labels_p)
        # for u,v,r in zip(u_nodes_p,v_nodes_p,ratings_p):
        #     if labels_p[u,v] == -1:
        #         labels_p[u,v] = rating_dict_p[r]
        #         # labels_p[u,v] = 1
        #     else:
        #         pass
        
        infile = open(os.path.join('data/',dataset,'gene.in'),'w')
        infile.write('%d\t%d\n'%(num_gene+num_drug+num_disease,357851))
        rating_mx_gg = np.full((num_gene, num_gene), neutral_rating, dtype=np.int32)
        for i in data_DGD_gg[['sourceid','relation','targetid']].values:
            rating_mx_gg[gene_dict[int(i[0])],gene_dict[int(i[2])]] = rating_dict_p[i[1]]
            infile.write('%s\t%s\n'%(gene_dict[int(i[0])],gene_dict[int(i[2])]))
        rating_mx_gg = sp.csr_matrix(rating_mx_gg+1,dtype=np.float32)

        rating_mx_cg = np.full((num_drug, num_gene), neutral_rating, dtype=np.int32)
        cg = data_DGD_cg[['sourceid','relation','targetid']].append(che_gene)
        for i in cg.values:
            c = i[2]
            g = i[0]
            if contain_alpha(i[0][0]):
                c = i[0]
                g = i[2]
            if c in drug_dict.keys():
                rating_mx_cg[drug_dict[c],gene_dict[int(g)]] = rating_dict_p[i[1]]
                infile.write('%s\t%s\n'%(drug_dict[c]+num_gene,gene_dict[int(g)]))
        rating_mx_cg = sp.csr_matrix(rating_mx_cg+1) 

        rating_mx_dg = np.full((num_disease, num_gene), neutral_rating, dtype=np.int32)
        dg = data_DGD_dg[['sourceid','relation','targetid']].append(dis_gene)

        for i in dg.values:
            d = i[0]
            g = i[2]
            if contain_alpha(i[2][0]):
                d = i[2]
                g = i[0]
            if d in disease_dict.keys():
                rating_mx_dg[disease_dict[d],gene_dict[int(g)]] = rating_dict_p[i[1]]
                infile.write('%s\t%s\n'%(disease_dict[d]+num_gene+num_drug,gene_dict[int(g)]))
        rating_mx_dg = sp.csr_matrix(rating_mx_dg+1) 
        infile.close()


        
  



        # for i in 

        # rating_mx_forsub = sp.csr_matrix(labels_p+1)

        if use_features == False:
            u_features = None
            v_features = None
            rel_features = None
            gene_features = None
            drug_sim = None
            dis_sim = None
        if use_features == True:
            if dataset == 'DATASET1':
                drug_sim = np.loadtxt('data/DATASET1/drug_sim.csv',delimiter=',',dtype=np.float32)
                dis_sim = np.loadtxt('data/DATASET1/dis_sim.csv',delimiter=',',dtype=np.float32)
                drugsim_dict = {j:i for i,j in enumerate(list(mesh2drugbank.values()))}
                lagcn_disease = pd.read_csv(
                    'data/DATASET1/disease.csv', usecols=['disease_id']
                )
                dissim_dict = {j:i for i,j in enumerate(lagcn_disease['disease_id'])}
                
                drugsimidx = list(drugsim_dict.keys())
                uid = [drugsim_dict[i] for i in list(u_dict.keys())]
                vid = [dissim_dict[i] for i in list(v_dict.keys())]
                drug_sim = drug_sim[uid]
                dis_sim = dis_sim[vid]
                embeddingfile = f'{dataset}/transe_embeddings/DGD+CTD_new_all/'
            if dataset == 'DATASET2':
                drug_drug = pd.read_csv('data/DATASET2/interactions/drug_drug.csv',delimiter=',',dtype=np.float32)
                drug_sim = np.zeros((num_drug,num_drug))
                drug_sim[drug_drug['Drug1'].astype(int).values,drug_drug['Drug2'].astype(int).values] = drug_drug['Sim'].values
                disease_disease = pd.read_csv('data/DATASET2/interactions/disease_disease.csv',delimiter=',',dtype=np.float32)
                dis_sim = np.zeros((num_disease,num_disease))
                dis_sim[disease_disease['Disease1'].astype(int).values,disease_disease['Disease2'].astype(int).values] = disease_disease['Sim'].values
                embeddingfile = f'{dataset}/transe_embeddings/'
            
            embedding = json.load(open(f'data/{embeddingfile}/embed.vec'))
            embedding_node = np.array(embedding['ent_embeddings.weight'])
            embedding_rel = np.array(embedding['rel_embeddings.weight'])
            name2id = {i.strip().split()[1]:i.strip().split()[0] for i in open(f'data/{embeddingfile}/entity2id.txt')}
            relname2id = {i.strip().split()[1]:i.strip().split()[0] for i in open(f'data/{embeddingfile}/relation2id.txt')}
            u_features = np.zeros((len(u_dict),embedding_node.shape[1]))
            idxx1 = [name2id[i] for i in u_dict.keys()]
            idxx1 = np.array(idxx1,dtype=int)
            u_features[list(u_dict.values())] = embedding_node[idxx1]

            v_features = np.zeros((len(v_dict),embedding_node.shape[1]))
            idxx2 = [name2id[i] for i in v_dict.keys()]
            idxx2 = np.array(idxx2,dtype=int)
            v_features[list(v_dict.values())] = embedding_node[idxx2]
            rel_features = np.zeros((len(rating_dict_p),embedding_rel.shape[1]))
            idxx3 = [relname2id[i] for i in  rating_dict_p.keys()]
            idxx3 = np.array(idxx3,dtype=int)
            rel_features[list(rating_dict_p.values())] = embedding_rel[idxx3]
            
            idxx4 = [int(name2id[str(i)]) for i in gene_dict.keys()]
            gene_features = np.zeros((len(gene_dict),embedding_node.shape[1]))
            gene_features[list(gene_dict.values())] = embedding_node[idxx4]

        # rating_mx_train = rating_mx_forsub
        class_values = np.sort(np.array(list(rating_dict.values())))
    
        rating_mx = rating_mx_train,rating_mx_gg,rating_mx_cg,rating_mx_dg
        json.dump(gene_dict,open(f'subgraph/{dataset}/gene_dict.json','w'))
        json.dump(drug_dict,open(f'subgraph/{dataset}/drug_dict.json','w'))
        json.dump(disease_dict,open(f'subgraph/{dataset}/disease_dict.json','w'))
        json.dump(rating_dict_p,open(f'subgraph/{dataset}/relation_dict.json','w'))

        dic={}
        for item in gene_dict.items():
            dic[str(item[0])]=item[1]
        for item in drug_dict.items():
            dic[item[0]]=item[1]+len(gene_dict)
        for item in disease_dict.items():
            dic[item[0]]=item[1]+len(gene_dict)+len(drug_dict)
        json.dump(dic,open(f'subgraph/{dataset}/allnode_dict.json','w'))


        
        # if os.path.exists("gene_%d_%d_merw.txt"%(per_path, walk_len)):
        #     gene_path = pd.read_csv("gene_%d_%d_merw.txt"%(per_path, walk_len))
        # else:
        # if os.path.exists(f"data/{dataset}/gene_%d_%d_merw.txt"%(per_path, walk_len)):
        #     print("loading path file...")
        #     gene_path = np.array([i.strip().split(' ')[:4] for i in open(f"data/{dataset}/gene_%d_%d_merw.txt"%(per_path, walk_len))]).reshape(-1,per_path,walk_len)
        #     # gene_path = np.loadtxt(f"data/{dataset}/gene_%d_%d_merw.txt"%(per_path, walk_len),dtype=int).reshape(-1,per_path,walk_len*2)
        # else:

        print("calculate path ...")
        # idconvert(f'data/{dataset}/convert_gene.in',f'data/{dataset}/gene.in',dic)
        cmd = './rw gene %d %d %d %s'%(per_path, walk_len, len(gene_dict),dataset)
        res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
        gene_path = res[1].decode()
        gene_path = [list(map(int, i.split()[:walk_len])) for i in gene_path.split('\n')[1:]][:-1]
        # gene_path = gene_path[:,:,:int(walk_len)]
        gene_path = np.array(gene_path,dtype=int).reshape(-1,per_path,walk_len)

        node_features = np.concatenate((gene_features,u_features,v_features))
        # gene_path = node_features[gene_path[:,:,:int(gene_path.shape[2]/2)]]
        

        paths = gene_path.reshape((1,-1,gene_path.shape[-2],gene_path.shape[-1]))
        paths = paths.swapaxes(0,1)
        paths = paths.reshape(paths.shape[0],-1,paths.shape[-1])
        ###random complete
        
        for i in range(walk_len):
            if len(paths[:,:,i][paths[:,:,i]==-1]) !=0:
                paths[:,:,i][paths[:,:,i]==-1] = paths[paths[:,:,i]==-1][:,0]

        # NAsize = paths[paths==-1].shape[0]
        # repalcenum = np.random.randint(0,len(gene_dict),(NAsize,))
        # paths[paths==-1] = repalcenum
        # while len(np.where(paths[:,:,0]==-1)[1]) != 0:
        #     idx = np.array(np.where(paths[:,:,0]==-1))
        #     paths[idx[0],idx[1]] = paths[idx[0],idx[1]-1]
        # paths = node_features[paths]
        # path_distance = gene_path[:,int(gene_path.shape[1]/2):]
        rel_map = data_DGD_gg.append(cg).append(dg)[['sourceid','relation','targetid']]
        reldic = {(str(s),str(t)):str(r) for s,r,t in rel_map.values}
        reldic.update({(str(t),str(s)):str(r) for s,r,t in rel_map.values if contain_alpha(str(s)+str(t))})
        dic_id2name={str(v):k for k,v in dic.items()}
        rating_dict_p.update({'NA':len(rating_dict_p)})
        rel_features = np.insert(rel_features,len(rel_features),0,axis=0)
        def maprel(paths):
            path_rel = []
            for node in paths:
                flag = 0
                node = np.insert(node,[1,2,3],[0,0,0],axis=1)
                while flag < walk_len-1:
                    find = [(dic_id2name[str(i)],dic_id2name[str(j)]) for i,j in node[:,[flag*2,flag*2+2]]]
                    rel = [rating_dict_p[reldic[i]]+num_disease+num_drug+num_gene if i in reldic.keys() else rating_dict_p['NA']+num_disease+num_drug+num_gene for i in find]
                    node[:,flag*2+1] = rel
                    flag += 1
                path_rel.append(node)
            return np.array(path_rel)
        paths = maprel(paths)

    # return u_features, v_features, rel_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
    return u_features, v_features, rel_features, gene_features, rating_mx, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, drug_dict, disease_dict,paths,node_features,drug_sim,dis_sim





def get_metrics(real_score, predict_score):
    real_score = np.array(real_score)
    predict_score = np.array(predict_score)
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    resultlist = [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
    resultlist = [float("%.4f"%i) for i in resultlist]
    return resultlist

def idconvert(mfile,ofile,dic):
    # dic = json.load(open(dic))
    dic = {str(key):str(value) for key,value in dic.items()}
    wf = open(ofile,'w')
    with open(mfile,'r') as rf:
        fistline = next(rf)
        wf.write(fistline)
        for line in rf:
            lines = line.strip().split()
            if lines[0]!=lines[1]:
                # new_lines = [dic[lines[0]],dic[lines[1]],lines[2]]
                new_lines = [dic[lines[0]],dic[lines[1]]]
                wf.write(' '.join(new_lines)+'\n')
    wf.close()

def totranse(all_nodes,all_relations,all_triples):
    # all_triples = pd.concat([data_DGD,che_gene,dis_gene,data_dr_di])
    # all_triples = all_triples.astype(str)
    # all_relations = all_triples['relation'].drop_duplicates()
    # all_nodes = pd.concat([all_triples['sourceid'],all_triples['targetid']]).drop_duplicates()
    with open('entity2id.txt','w') as wf:
        entity2id={}
        wf.write(f'{len(all_nodes)}\n')
        for idx,i in enumerate(all_nodes):
            entity2id[i]=idx
            wf.write(f'{idx}\t{i}\n')
    with open('relation2id.txt','w') as wf:
        relation2id = {}
        wf.write(f'{len(all_relations)}\n')
        for idx,i in enumerate(all_relations):
            relation2id[i]=idx
            wf.write(f'{idx}\t{i}\n')
    with open('train2id.txt','w') as wf:
        wf.write(f'{len(all_triples)}\n')
        for i in all_triples:
            wf.write(f"{entity2id[i[0]]}\t{entity2id[i[2]]}\t{relation2id[i[1]]}\n")


def process_pharmkg(df):
    df.relation[df['relation'].isin(['I','Iw'])]='I'
    df.relation[df['relation'].isin(['Te','D','X'])]='Te'
    df = df.drop_duplicates(subset=['sourceid','targetid','relation'])
    return df

def priority(df,dftype):
    priority = {'gg':['Ra','Q','Rg','B','E','GG','I'],'cg':['A','N','O','K','Z','B','E','I'],'dg':['P','Te','U','ML','I']}
    relations = {i:j for j,i in enumerate(priority[dftype])}
    df = df[df['relation'].isin(priority[dftype])]
    rel_priority = [relations[i] for i in df['relation']]
    df.insert(loc=3,column='rel_priority',value=rel_priority)
    df = df.sort_values(by=['rel_priority'])
    df = df.drop_duplicates(subset=['sourceid','targetid'],keep='first')
    return df

def CTDintegrate(df):
    relationmap = {}
    with open('data/CTDrelation.txt') as f:
        for line in f:
            lines = line.strip().split('\t')
            relationmap[lines[0]] = lines[1]
    rel = [relationmap[i.split('|')[0].split('^')[1]] for i in df['relation']]
    df.relation = rel
    df = df[df['relation']!='remove']
    df = priority(df,'cg')
    return df

import re
def contain_alpha(str):
  my_re = re.compile(r'[A-Za-z]',re.S)
  res = re.findall(my_re,str)
  if len(res):
      return True
  else:
      return False
