
import math
import multiprocessing as mp
import os
import time

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
from utils import get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_multiple_epochs(train_dataset,
                          test_dataset,
                          paths,
                          model,
                          epochs,
                          batch_size,
                          lr,
                          lr_decay_factor,
                          lr_decay_step_size,
                          weight_decay,
                          pos_weight,
                          test_freq=1, 
                          logger=None, 
                          continue_from=None, 
                          res_dir=None,
                          per_path=None
                          ):

    metrics = []
    best_result = [0]
    flag = 0
    if train_dataset.__class__.__name__ == 'MyDynamicDataset':
        num_workers = mp.cpu_count()
    else:
        num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    if test_dataset.__class__.__name__ == 'MyDynamicDataset':
        num_workers = mp.cpu_count()
    else:
        num_workers = 2
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, 
                             num_workers=num_workers)

    model.to(device)
    # model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 1
    if continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(res_dir, 'model_checkpoint{}.pth'.format(continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(continue_from)))
        )
        start_epoch = continue_from + 1
        epochs -= continue_from

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    batch_pbar = len(train_dataset) < 100000
    t_start = time.perf_counter()
    if not batch_pbar:
        pbar = tqdm(range(start_epoch, epochs + start_epoch))
    else:
        pbar = range(start_epoch, epochs + start_epoch)
    for n_iter,epoch in enumerate(pbar):

        onepaths = paths
        train_loss,(h_n,gcn),x_all = train(model, onepaths, optimizer, train_loader, device, pos_weight, regression=True,  
                           show_progress=False, epoch=n_iter)

        if epoch % test_freq == 0:
            metric,fullresult = eval_metric(model, onepaths, test_loader, device, pos_weight, show_progress=False)
            # metric,fullresult = eval_metric(model, test_loader, device, show_progress=batch_pbar)
            metrics.append(metric)
            if metric >= best_result[0]:
                best_result = fullresult
                flag = 0
            else:
                flag += 1
        else:
            metrics.append(np.nan)
            fullresult = 0
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_metric': metrics[-1],
        }
        if epoch % test_freq == 0:
            if not batch_pbar:
                pbar.set_description(
                    'Epoch {}, train loss {:.6f}, test metric {:.6f}'.format(*eval_info.values())
                )
                print(fullresult) 
            else:
                print('Epoch {}, train loss {:.6f}, test metric {:.6f}'.format(*eval_info.values()))
                print(fullresult) 
                print("==="*10)


        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if logger is not None:
            logger(eval_info, model, optimizer)
    print("the best result:")
    print(best_result)
        #     break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return metrics[-1]


def test_once(test_dataset,
              paths,
              model,
              batch_size,
              pos_weight,
              logger=None, 
              ensemble=False, 
              checkpoints=None,
              per_path=None):

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    t_start = time.perf_counter()
    onepaths = paths
    if ensemble and checkpoints:
        metric,fullresult = eval_metric_ensemble(model, checkpoints, test_loader, device, pos_weight, show_progress=True)
    else:
        metric,fullresult = eval_metric(model, onepaths,test_loader, device, pos_weight, show_progress=True)
    t_end = time.perf_counter()
    duration = t_end - t_start
    print('Test Once Metric: {:.6f}, Duration: {:.6f}'.format(metric, duration))
    print(fullresult)
    epoch_info = 'test_once' if not ensemble else 'ensemble'
    eval_info = {
        'epoch': epoch_info,
        'train_loss': 0,
        'test_metric': metric,
        }
    if logger is not None:
        logger(eval_info, None, None)
    return metric


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, paths, optimizer, loader, device, pos_weight, regression=True, 
          show_progress=False, epoch=None):
    model.train()
    total_loss = 0
    if show_progress:
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        data.paths = torch.LongTensor(paths)
        data = data.to(device)
        out,lstm_emb,gcn_emb = model(data)
        if regression:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight)
            loss = criterion(out.view(-1), data.y.view(-1))
            
        else:
            loss = F.nll_loss(out, data.y.view(-1))
        if show_progress:
            pbar.set_description('Epoch {}, batch loss: {}'.format(epoch, loss.item()))

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    # return total_loss / len(loader.dataset)
    return loss,lstm_emb,gcn_emb

def ceil(y_preds):
    return -1 if y_preds < 0 else 1

def eval_loss(model, paths,loader, device, pos_weight, regression=True, show_progress=False):
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader

    Rs = []
    Ys = []
    # paths = paths.reshape((1000,-1,paths.shape[-2],paths.shape[-1]))
    for data in pbar:
        data.paths = torch.LongTensor(paths)
        data = data.to(device)

        with torch.no_grad():
            out,_,_ = model(data)

        y = data.y
        Rs.extend(out.detach().view(-1).tolist())
        Ys.extend(y.view(-1).tolist())

        if regression:
            # loss += F.mse_loss(out, data.y.view(-1), reduction='sum').item()
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight)
            loss += criterion(out.view(-1), data.y.view(-1)).item()
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()

    if regression:
        metrics = get_metrics(Ys,Rs)
        metric = metrics[0]
        return metric,metrics
    else:
        return loss / len(loader.dataset)


def eval_metric(model, paths,loader, device, pos_weight, show_regression=False, show_progress=False):
    loss,fullresult = eval_loss(model, paths, loader, device, pos_weight, True, show_progress)
    if show_regression:
        rmse = math.sqrt(loss)
        return rmse
    else:
        return loss,fullresult


def eval_loss_ensemble(model, checkpoints, loader, device, pos_weight, regression=True, show_progress=False):
    loss = 0
    Outs = []
    ys = []
    for i, checkpoint in enumerate(checkpoints):
        if show_progress:
            print('Testing begins...')
            pbar = tqdm(loader)
        else:
            pbar = loader
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        outs = []
        for data in pbar:
            data = data.to(device)
            if i == 0:
                ys.append(data.y.view(-1))
            with torch.no_grad():
                out,_,_ = model(data)
                outs.append(out)
        if i == 0:
            ys = torch.cat(ys, 0)
        outs = torch.cat(outs, 0).view(-1, 1)
        Outs.append(outs)
    Outs = torch.cat(Outs, 1).mean(1)
    if regression:
        # loss += F.mse_loss(Outs, ys, reduction='sum').item()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight)
        loss += criterion(Outs, ys).item()
        
    else:
        loss += F.nll_loss(Outs, ys, reduction='sum').item()
    torch.cuda.empty_cache()

    if regression:
        metrics = get_metrics(ys,Outs)
        metric = metrics[0]
        fullresult = metrics
        return metric,fullresult
    else:
        return loss / len(loader.dataset)



def eval_metric_ensemble(model, checkpoints, loader, device, pos_weight, show_regression=False, show_progress=False):
    loss,fullresult = eval_loss_ensemble(model, checkpoints, loader, device, pos_weight, True, show_progress)
    if show_regression:
        rmse = math.sqrt(loss)
        return rmse
    else:
        return loss,fullresult


def predict(model, graphs, paths, res_dir, data_name, class_values, checkpoints=None, ensemble=False, num=20, sort_by='prediction',per_path =None):
    onepaths = paths
    if ensemble:
        models = []
        for checkpoint in checkpoints:
            model.load_state_dict(torch.load(checkpoint))
            models.append(model)
    else:
        models = [model]

    all_R = []
    for model in models:
        model.eval()
        model.to(device)

        R = []
        graph_loader = DataLoader(graphs, 1, shuffle=False)
        for data in tqdm(graph_loader):
            data.paths = torch.Tensor(onepaths)
            data = data.to(device)
            r = model(data)[0].detach()
            R.extend(r.view(-1).tolist())
        all_R.append(R)

    avg_R = [np.mean(e) for e in zip(*all_R)]
    res_path = os.path.join(res_dir, "predictions_{}_{}_full.csv".format(data_name, num))
    res = pd.read_csv(res_path)
    res['prediction'] = avg_R

    res_path_select = os.path.join(res_dir, "predictions_{}_{}.csv".format(data_name, num))
    if sort_by == 'prediction':
        order = np.argsort(R).tolist()
        highest = order[-num:]
        lowest = order[:num]
        select = res.loc[highest+lowest, :]
    elif sort_by == 'random':  
        order = np.random.permutation(range(len(R))).tolist()
        select = res.loc[order[num*2:], :]

    res.to_csv(
        res_path,
        index=False
    )
    select.to_csv(
        res_path_select,
        index=False
    )

def save_test_results(model, graphs, res_dir, data_name, mode='test'):
    model.eval()
    model.to(device)
    
    R = []
    Y = []
    graph_loader = DataLoader(graphs, 50, shuffle=False)
    for data in tqdm(graph_loader):
        data = data.to(device)
        r = model(data)[0].detach()
        y = data.y
        R.extend(r.view(-1).tolist())
        Y.extend(y.view(-1).tolist())
    
    res = pd.DataFrame({'Y': Y, 'R':R})
    res_path = os.path.join(res_dir, f"{mode}_predictions_{data_name}.csv")
    res.to_csv(res_path, index=False)


    def get_preds(row):
        if data_name == 'DGIdb':
            return round(row)
        else:
            return -1 if row < 0 else 1

    metrics = get_metrics(res['Y'],res['R'])
    accuracy = metrics[3]

    print('Final Test Accuracy: {:.6f}'.
          format(accuracy))
    print('fullresult')
    print(metrics)


import torch.utils.data
from torch._six import container_abcs, int_classes, string_classes
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data


class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch), **kwargs)


class DataListLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a python list.

    .. note::

        This data loader should be used for multi-gpu support via
        :class:`torch_geometric.nn.DataParallel`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataListLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=lambda data_list: data_list, **kwargs)


class DenseCollater(object):
    def collate(self, data_list):
        batch = Batch()
        for key in data_list[0].keys:
            batch[key] = default_collate([d[key] for d in data_list])
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class DenseDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    .. note::

        To make use of this data loader, all graphs in the dataset needs to
        have the same shape for each its attributes.
        Therefore, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DenseDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=DenseCollater(), **kwargs)
