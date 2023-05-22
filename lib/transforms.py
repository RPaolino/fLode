from lib.utils import *
from scipy.sparse import coo_matrix
import time
import torch
from torch_scatter import scatter_sum
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.data.datapipes import functional_transform

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.utils.extmath import randomized_svd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transform(normalize_features, norm_ord, norm_dim, undirected, self_loops, lcc, sparsity, sklearn, verbose=False):
    transform_list=[]
    if normalize_features:
        transform_list += [CustomNormalizeFeatures(ord=norm_ord, dim=norm_dim)] 
    transform_list += [CustomSelfLoops(value=self_loops)]
    if undirected:
        transform_list += [T.ToUndirected()]
    if lcc:
        transform_list += [T.LargestConnectedComponents(num_components=1, connection='weak')]
    transform_list += [SNA(sparsity=sparsity, sklearn=sklearn), SNL()]
    if verbose:
        print("Transforms")
        for t in transform_list: 
            print(f'| {t}')
    return T.Compose(transform_list)

@functional_transform('custom_normalize_features')
class CustomNormalizeFeatures(T.BaseTransform):
    r"""
    Normalizes the features w.r.t. the specified dim and the specified p-norm.
    If you want the default NormalizeFeatures from torch_geometric.transform,
    put ord="sum" and dim=-1.
    Args:
        ord: p-norm that is computed.
        dim: along which dimension to compute the p-norm
    """
    def __init__(self, ord, dim):
        super().__init__()
        self.ord=ord
        self.dim=dim

    def __call__(self, data):
        if self.ord=="sum":
            data.x = data.x - data.x.min()
            data.x.div_(data.x.sum(dim=self.dim, keepdim=True).clamp_(min=1.))
        else:
            norm = torch.linalg.norm(data.x, ord=self.ord, dim=self.dim, keepdim=True).clamp_(min=1.)
            data.x.div_(norm)
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(ord={self.ord}, dim={self.dim})'

@functional_transform('custom_self_loops')
class CustomSelfLoops(T.BaseTransform):
    r"""
    Compute the symmetrically normalized Laplacian. Useful to compute the Dirichlet energy.
    """
    def __init__(self, value):
        self.value=value
        
    def __call__(self, data):
        if "edge_weight" not in data:
            edge_weight = torch.ones(data.edge_index.shape[1])
        else:
            edge_weight = data.edge_weight
        # self_loops = (data.edge_index[0]==data.edge_index[1])
        # edge_weight[self_loops] = self.value
        data.edge_index, data.edge_weight = add_remaining_self_loops(
            edge_index=data.edge_index, 
            edge_attr=edge_weight, 
            fill_value=self.value, 
            num_nodes=data.num_nodes
        )
        return data

@functional_transform('sna')
class SNA(T.BaseTransform):
    r"""
    Compute the symmetrically normalized adjacency and its singular value decomposition.
    Args:
        sparsity (float, default 0): how many singular values to consider
        sklearn (bool, default False): if set to True, uses the scikit-learn-intelex ext_math library to compute svd.
    """
    def __init__(self, sparsity= 0.0, sklearn: bool=False):
        self.sparsity=sparsity
        self.sklearn=sklearn
        
    def __call__(self, data):
        
        row, col = data.edge_index[0], data.edge_index[1]
        
        if "edge_weight" not in data:
            edge_weight = torch.ones(data.edge_index.shape[1])
        else:
            edge_weight = data.edge_weight
        
        in_deg = scatter_sum(edge_weight, row, dim=0, dim_size=data.num_nodes)       
        in_deg_inv_sqrt = in_deg.pow(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt==float('inf'), 0.)
        
        out_deg = scatter_sum(edge_weight, col, dim=0, dim_size=data.num_nodes)
        out_deg_inv_sqrt = out_deg.pow(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt==float('inf'), 0.)
        
        edge_weight =  in_deg_inv_sqrt[row] * edge_weight * out_deg_inv_sqrt[col]
        
        data.sna = torch.sparse_coo_tensor(indices=data.edge_index, values=edge_weight, size=(data.num_nodes, data.num_nodes))    
        
        if type(self.sparsity)==float:
            n_components=int(np.ceil((1.-self.sparsity)*data.num_nodes))
        elif type(self.sparsity)==int:
            n_components=self.sparsity
        initial_time = time.time()
        if self.sklearn:
            sp_sna = coo_matrix((edge_weight, data.edge_index), shape=(data.num_nodes, data.num_nodes))
            U, S, Vh = randomized_svd(sp_sna, n_components=n_components)
            data.U, data.S, data.Vh = torch.from_numpy(U).to(device), torch.from_numpy(S).to(device), torch.from_numpy(Vh).to(device)
        else:
            data.U, data.S, data.Vh = torch.linalg.svd(data.sna.to_dense())
            data.U = data.U[:, :n_components]
            data.S = data.S[:n_components]
            data.Vh = data.Vh[:n_components, :]
        print(f'SVD elapsed time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-initial_time))}')
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sparsity={self.sparsity}, sklearn={self.sklearn})'
    
@functional_transform('snl')
class SNL(T.BaseTransform):
    r"""
    Compute the symmetrically normalized Laplacian. Useful to compute the Dirichlet energy.
    """
    def __call__(self, data):
        
        if "edge_weight" not in data:
            edge_weight = torch.ones(data.edge_index.shape[1])
        else:
            edge_weight = data.edge_weight
            
        edge_index, edge_weight = add_remaining_self_loops(data.edge_index, edge_attr=edge_weight, fill_value=0.)
                    
        row, col = edge_index[0], edge_index[1]
        self_loops = (row==col)
        
        in_deg = scatter_sum(edge_weight, row, dim=0, dim_size=data.num_nodes)
        in_deg_inv_sqrt = in_deg.pow(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt==float('inf'), 0.)
        
        out_deg = scatter_sum(edge_weight, col, dim=0, dim_size=data.num_nodes)
        out_deg_inv_sqrt = out_deg.pow(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt==float('inf'), 0.)
        
        edge_weight =  -in_deg_inv_sqrt[row] * edge_weight * out_deg_inv_sqrt[col]
        edge_weight[self_loops] = edge_weight[self_loops]+1
        
        data.snl=torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=(data.num_nodes, data.num_nodes))       
        
        return data
        
        