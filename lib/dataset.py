from lib.transforms import *
from lib.utils import *
import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor


from sklearn.model_selection import train_test_split

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
        dataset = Planetoid(root='data', split="geom-gcn", name=dataset_name, transform=transform)
    elif dataset_name == "film":
        dataset =  Actor(root='data', transform=transform)
    elif dataset_name in ["chameleon", "squirrel", "crocodile"]:
        dataset =  WikipediaNetwork(root='data', name=dataset_name, transform=transform)
    elif dataset_name=="dsbm":
        node = 500
        cluster = 5
        p_q = 0.85
        sizes = [node]*cluster

        p_in, p_inter = 0.1, 0.1
        prob = np.diag([p_in]*cluster)
        prob[prob == 0] = p_inter
        for seed in [10]:
            _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=seed)
            data = to_dataset(A, label, save_path = '../../dataset/data/tmp/syn/syn2Seed'+str(seed)+'.pk')
        dataset = [transform(data)]
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
    return dataset, data
  
def desymmetric_stochastic(sizes = [100, 100, 100],
                probs = [[0.5, 0.45, 0.45],
                         [0.45, 0.5, 0.45],
                         [0.45, 0.45, 0.5]],
                seed = 0,
                off_diag_prob = 0.9, 
                norm = False):
    #from sklearn.model_selection import train_test_split
    
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    original_A = nx.adjacency_matrix(g).todense()
    A = original_A.copy()
    
    # for blocks represent adj within clusters
    accum_size = 0
    for s in sizes:
        x, y = np.where(np.triu(original_A[accum_size:s+accum_size,accum_size:s+accum_size]))
        x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)
        A[x1+accum_size, y1+accum_size] = 0
        A[y2+accum_size, x2+accum_size] = 0
        accum_size += s

    # for blocks represent adj out of clusters (cluster2cluster edges)
    accum_x, accum_y = 0, 0
    n_cluster = len(sizes)
    
    for i in range(n_cluster):
        accum_y = accum_x + sizes[i]
        for j in range(i+1, n_cluster):
            x, y = np.where(original_A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y])
            x1, x2, y1, y2 = train_test_split(x, y, test_size=off_diag_prob)
            
            A[x1+accum_x, y1+accum_y] = 0
            A[y2+accum_y, x2+accum_x] = 0
                
            accum_y += sizes[j]
            
        accum_x += sizes[i]
    # label assignment based on parameter sizes 
    label = []
    for i, s in enumerate(sizes):
        label.extend([i]*s)
    label = np.array(label)      

    return np.array(original_A), np.array(A), label

def to_dataset(A, label, save_path):
    import pickle as pk
    from numpy import linalg as LA
    from torch_geometric.data import Data
    from scipy import sparse

    masks = {}
    masks['train'], masks['val'], masks['test'] = [], [] , []
    for split in range(10):
        mask = train_test_split_magnet(label, seed=split, train_examples_per_class=10, val_size=500, test_size=None)
        mask['train'] = torch.from_numpy(mask['train']).bool()
        mask['val'] = torch.from_numpy(mask['val']).bool()
        mask['test'] = torch.from_numpy(mask['test']).bool()
    
        masks['train'].append(mask['train'].unsqueeze(-1))
        masks['val'].append(mask['val'].unsqueeze(-1))
        masks['test'].append(mask['test'].unsqueeze(-1))
    
    label = torch.from_numpy(label).long()

    s_A = sparse.csr_matrix(A)
    coo = s_A.tocoo()
    values = coo.data
    
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    '''
    A_sym = 0.5*(A + A.T)
    A_sym[A_sym > 0] = 1
    d_out = np.sum(np.array(A_sym), axis = 1)
    _, v = LA.eigh(d_out - A_sym)
    features = torch.from_numpy(np.sum(v, axis = 1, keepdims = True)).float()
    '''
    s = np.random.normal(0, 1.0, (len(A), 1))
    features = torch.from_numpy(s).float()

    data = Data(x=features, edge_index=indices, edge_weight=None, y=label)
    data.train_mask = torch.cat(masks['train'], axis=-1) 
    data.val_mask   = torch.cat(masks['val'], axis=-1)
    data.test_mask  = torch.cat(masks['test'], axis=-1)

    return data

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def train_test_split_magnet(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])
