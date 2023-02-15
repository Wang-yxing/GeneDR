import os
import sys

import numpy as np
import torch

os.chdir(sys.path[0])
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import json
import os.path
from shutil import copy, copytree, rmtree

from dataset import *
from models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train import *
# from parser import get_basic_configs
from utils import load_data_from_database


def logger(info, model, optimizer):
    epoch, train_loss, test_metric = info['epoch'], info['train_loss'], info['test_metric']
    with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
        f.write('Epoch {}, train loss {:.4f}, test metric {:.6f}\n'.format(
            epoch, train_loss, test_metric))
    if type(epoch) == int and epoch % args.save_interval == 0:
        print('Saving model states...')
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(
            args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
        )
        if model is not None:
            torch.save(model.state_dict(), model_name)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), optimizer_name)


class convert:
    def __init__(self, **entries):
        self.__dict__.update(entries)
args = json.load(open('args.json'))
args = convert(**args)

args.nums = 0
rating_map, post_rating_map = None, None
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.testing:
    val_test_appendix = 'testmode'
else:
    val_test_appendix = 'valmode'
args.res_dir = os.path.join(
    args.file_dir, 'outputs/{}_{}_{}'.format(
        args.data_name, args.save_appendix, val_test_appendix
    )
)
args.model_pos = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(args.epochs))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
os.system(f'cp *.py {args.res_dir}/')
os.system(f'cp *.sh {args.res_dir}/')
os.system(f'cp *.json {args.res_dir}/')
print('Python files: *.py, *.json and *.sh is saved.')


(
    u_features, v_features, rel_features, gene_features, adj_train, train_labels, train_u_indices, train_v_indices,
    val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, 
    test_v_indices, class_values, u_dict, v_dict, gene_path,node_features,drug_sim,dis_sim
) = load_data_from_database(dataset=args.dataset,mode=args.mode, testing=args.testing, rating_map=rating_map,use_features=args.use_features,with_addds=args.with_pharmkg,posneg_ratio=args.posneg_ratio,per_path=args.per_path,walk_len=args.walk_len)

print('All ratings are:')
print(class_values)
'''
Explanations of the above preprocessing:
    class_values are all the original continuous ratings, e.g. 0.5, 2...
    They are transformed to rating labels 0, 1, 2... acsendingly.
    Thus, to get the original rating from a rating label, apply: class_values[label]
    Note that train_labels etc. are all rating labels.
    But the numbers in adj_train are rating labels + 1, why? Because to accomodate 
    neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
    If testing=True, adj_train will include both train and val ratings, and all train 
    data will be the combination of train and val.
'''

number_of_u, number_of_v = len(u_dict), len(v_dict)
if args.use_features:
    # # u_features, v_features = u_features.toarray(), v_features.toarray()
    # n_features = u_features.shape[1] + v_features.shape[1]
    n_features = u_features.shape[1] + v_features.shape[1]
    print('Number of user features {}, item features {}, total features {}'.format(
        u_features.shape[1], v_features.shape[1], n_features))
    features = (u_features,v_features, rel_features,gene_features)
else:
    u_features, v_features = None, None
    n_features = 0
    features = None



if args.debug:  # use a small number of data to debug
    num_data = 1000
    train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
    val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]
    test_u_indices, test_v_indices = test_u_indices[:num_data], test_v_indices[:num_data]

train_indices = (train_u_indices, train_v_indices)
val_indices = (val_u_indices, val_v_indices)
test_indices = (test_u_indices, test_v_indices)
print('#train: %d, #val: %d, #test: %d' % (
    len(train_u_indices), 
    len(val_u_indices), 
    len(test_u_indices), 
))

ALL = False
if args.probe:
    u2name = {v:k for k,v in u_dict.items()}
    v2name = {v:k for k,v in v_dict.items()}
    u_index, v_index = np.hstack([train_u_indices, test_u_indices]), np.hstack([train_v_indices, test_v_indices])
    exits_tuple = set(zip(u_index, v_index))
    if ALL:
        u_i = np.arange(number_of_u)
        v_i = np.arange(number_of_v)
        probe_u_idx = np.repeat(u_i, number_of_v)
        probe_v_idx = np.tile(v_i, number_of_u)
    else:
        u_i = np.unique(test_u_indices)
        v_i = np.arange(number_of_v)
        probe_u_idx = np.repeat(u_i, number_of_v)
        probe_v_idx = np.tile(v_i, len(u_i))
    probe_tuple = set(zip(probe_u_idx, probe_v_idx))

    probe_u_indices, probe_v_indices = [], []
    probe_list = list(probe_tuple - exits_tuple)
    for probe_pair in probe_list:
        probe_u_indices.append(probe_pair[0])
        probe_v_indices.append(probe_pair[1])

    probe_u_indices, probe_v_indices = np.array(probe_u_indices), np.array(probe_v_indices)
    probe_indices = (probe_u_indices, probe_v_indices)
    probe_labels = np.ones(len(probe_u_indices), dtype=int)

    res_path = os.path.join(args.res_dir, "predictions_{}_{}_full.csv".format(args.data_name, args.nums))
    drug_id = [u2name[i] for i in probe_u_indices]
    gene_id = [v2name[i] for i in probe_v_indices]
    res_df = pd.DataFrame({'Drug': drug_id, 'Gene': gene_id})
    res_df.to_csv(res_path, index=False)

    print('#probe: %d' % (
        len(probe_u_indices) 
    ))

train_graphs, val_graphs, test_graphs = None, None, None
data_combo = (args.data_name, args.data_appendix, val_test_appendix)

if args.reprocess:
    # if reprocess=True, delete the previously cached data and reprocess.
    if os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
        rmtree('data/{}{}/{}/train'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/val'.format(*data_combo)):
        rmtree('data/{}{}/{}/val'.format(*data_combo))
    if os.path.isdir('data/{}{}/{}/test'.format(*data_combo)):
        rmtree('data/{}{}/{}/test'.format(*data_combo))

paths = gene_path


dataset_class = 'MyDynamicDataset' if args.dynamic_train else 'MyDataset'
# dataset_class = 'MyDataset'
train_graphs = eval(dataset_class)(
    'data/{}{}/{}/train'.format(*data_combo),
    adj_train,
    train_indices, 
    train_labels, 
    args.hop, 
    args.sample_ratio, 
    args.max_nodes_per_hop, 
    # u_features, 
    # v_features,
    features,
    drug_sim,dis_sim,
    # rel_features, 
    class_values, 
    # gene_path,
    max_num=args.max_train_num
)

dataset_class = 'MyDynamicDataset' if args.dynamic_test else 'MyDataset'
# dataset_class = 'MyDataset'
test_graphs = eval(dataset_class)(
    'data/{}{}/{}/test'.format(*data_combo),
    adj_train,
    test_indices, 
    test_labels, 
    args.hop, 
    args.sample_ratio, 
    args.max_nodes_per_hop, 
    # u_features, 
    # v_features,
    features,
    drug_sim,dis_sim,
    # rel_features, 
    class_values, 
    # gene_path, 
    max_num=args.max_test_num
)
if not args.testing:
    dataset_class = 'MyDynamicDataset' if args.dynamic_val else 'MyDataset'
    # dataset_class = 'MyDataset'
    val_graphs = eval(dataset_class)(
        'data/{}{}/{}/val'.format(*data_combo),
        adj_train,
        val_indices, 
        val_labels, 
        args.hop, 
        args.sample_ratio, 
        args.max_nodes_per_hop, 
        # u_features, 
        # v_features,
        features,
        drug_sim,dis_sim,
        # rel_features, 
        class_values, 
        # gene_path, 
        max_num=args.max_val_num
    )

# Determine testing data (on which data to evaluate the trained model
if not args.testing: 
    test_graphs = val_graphs

print('Used #train graphs: %d, #test graphs: %d' % (
    len(train_graphs), 
    len(test_graphs), 
))


'''
    Train and apply the GNN model
'''
num_relations = len(class_values)
multiply_by = 1

pos_weight = torch.tensor([float((train_labels.shape[0]-train_labels.sum())/train_labels.sum())]).to(device)


model = GeneDR(
    train_graphs, 
    latent_dim=[args.hidden]*2, 
    num_relations=num_relations, 
    num_bases=4, 
    regression=True, 
    adj_dropout=args.adj_dropout, 
    force_undirected=args.force_undirected, 
    side_features=args.use_features, 
    n_side_features=n_features, 
    multiply_by=multiply_by,
    per_path=args.per_path
)

def main(model):
    if not args.no_train:
        train_multiple_epochs(
            train_graphs,
            test_graphs,
            paths,
            model,
            args.epochs, 
            args.batch_size, 
            args.lr, 
            lr_decay_factor=args.lr_decay_factor, 
            lr_decay_step_size=args.lr_decay_step_size, 
            weight_decay=0.0005, 
            ARR=args.ARR, 
            test_freq=args.test_freq, 
            logger=logger, 
            continue_from=args.continue_from, 
            res_dir=args.res_dir,
            pos_weight=pos_weight,
            per_path=args.per_path
        )



    if args.testing:
        model.load_state_dict(torch.load(args.model_pos))
        test_once(
            test_graphs,
            paths,
            model=model,
            batch_size=args.batch_size,
            pos_weight=pos_weight,
            logger=logger,
            per_path=args.per_path
        )

    if not args.testing:
        model.load_state_dict(torch.load(args.model_pos))
        test_once(
            test_graphs,
            paths,
            model=model,
            batch_size=args.batch_size,
            pos_weight=pos_weight,
            logger=logger,
            per_path=args.per_path
        )
        test_once(
            val_graphs,
            paths,
            model=model,
            batch_size=args.batch_size,
            pos_weight=pos_weight,
            logger=logger,
            per_path=args.per_path
        )


    if args.save_results:
        if args.ensemble:
            start_epoch, end_epoch, interval = args.epochs-30, args.epochs, 10

            checkpoints = [
                os.path.join(args.res_dir, 'model_checkpoint%d.pth' %x) 
                for x in range(start_epoch, end_epoch+1, interval)
            ]
            epoch_info = 'ensemble of range({}, {}, {})'.format(
                start_epoch, end_epoch, interval
            )

            for idx, checkpoint in enumerate(checkpoints):
                model.load_state_dict(torch.load(checkpoint))
                save_test_results(
                    model=model,
                    graphs=test_graphs,
                    res_dir=args.res_dir,
                    data_name=args.data_name+'_epoch'+str(idx*interval+start_epoch),
                    mode='test'
                )
        else:
            model.load_state_dict(torch.load(args.model_pos))
            save_test_results(
                model=model,
                graphs=test_graphs,
                res_dir=args.res_dir,
                data_name=args.data_name
            )



for i in [model]:
    main(i)
