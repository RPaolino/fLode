import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, HeterophilousGraphDataset

from lib.transforms import *
from lib.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataset(dataset_name, transform, verbose=False):
    r''''
    Upload the dataset.
    Args:
        dataset_name (str): the name of the dataset to upload. 
            Supported Cora, Citeseer, Pubmed, Actor, chameleon, squirrel, crocodile.
        transform (instance of torch_geometric.transforms.BaseTransform 
            or torch_geometric.transforms.Compose): transforms to apply to the dataset.
    '''
    if dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root='.data', split="geom-gcn", name=dataset_name, pre_transform=transform)
        metric_name = "acc"
    elif dataset_name == "film":
        dataset =  Actor(root='.data/film/geom-gcn', pre_transform=transform)
        metric_name = "acc"
    elif dataset_name in ["chameleon", "squirrel", "crocodile"]:
        dataset =  WikipediaNetwork(root='.data', name=dataset_name, pre_transform=transform)
        metric_name = "acc"
    elif dataset_name in ["Roman-empire", "Minesweeper", "Tolokers", "Amazon-ratings", "Questions"]:
        dataset =  HeterophilousGraphDataset(root='.data', name=dataset_name, pre_transform=transform)
        if dataset_name in ["Questions", "Minesweeper", "Tolokers"]:
            metric_name = "roc_auc"
        else:
            metric_name = "acc"
    else:
        assert False, f'Dataset {dataset_name} not implemented!'
    data=dataset[0].to(device)
    if verbose:
        # Gather some statistics about the graph.
        print(f'Dataset: {colortext(dataset_name, "c")}')
        print(f'| num. graphs: {len(dataset)}, num. features {dataset.num_features}, num. classes: {dataset.num_classes}')
        print(f'Data')
        print(f'| num. nodes {data.num_nodes}, isolated nodes: {data.has_isolated_nodes()}, self-loops: {data.has_self_loops()}, undirected: {data.is_undirected()}')
        print(f'| {data}')
    return dataset, data, metric_name