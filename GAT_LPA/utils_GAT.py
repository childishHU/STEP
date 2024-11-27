from scipy.spatial.distance import cdist
from typing import NoReturn
from sklearn.neighbors import NearestNeighbors 
from torch_sparse import SparseTensor
import scipy.sparse as sp
import numpy as np
import torch

class SimpleProgressBar:
    """
    Progress bar to display progress during estimation
    """

    def __init__(self,
                 max_value: int,
                 length: int = 20,
                 symbol: str = "=",
                 silent_mode: bool = False,
                 ) -> None:
        """
        :param max_value: int, total number of epochs to be used.
        :param length: int, number of markers to use.
        :param symbol: str, symbol to use as indicator.
        :param silent_mode: bool, whether to use silent mode, default is False.
        """

        self.symbol = symbol
        self.mx = max_value
        self.len = length
        self.delta = self.mx / self.len
        self.ndigits = len(str(self.mx))

        print("\r\n")

        if silent_mode:
            self.call_func = self._silent
        else:
            self.call_func = self._verbose

    def _verbose(self,
                 epoch: int,
                 value: float,
                 ) -> None:

        """
        :param epoch: int, current epoch
        :param value: float, value to display
        """

        progress = self.symbol * int((epoch / self.delta))
        print(f"\r"
              f"Epoch : {epoch + 1:<{self.ndigits}}/{self.mx:<{self.ndigits}}"
              f" | Loss : {value:9E}"
              f" | \x1b[1;37m["
              f" \x1b[0;36m{progress:<{self.len}}"
              f"\x1b[1;37m]"
              f" \x1b[0m",
              end="")

    def _silent(self,
                *args,
                **kwargs,
                ) -> NoReturn:
        pass

    def __call__(self,
                 epoch: int,
                 value: float,
                 ) -> NoReturn:

        self.call_func(epoch, value)

def construct_interaction(position, n_neighbors=10):
    """Constructing spot-to-spot interactive graph"""
    
    # calculate distance matrix
    distance_matrix = cdist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1  
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    return adj
    
def construct_interaction_KNN(position, n_neighbors=10):
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    return adj
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)