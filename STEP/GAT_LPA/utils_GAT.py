from typing import NoReturn

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from torch_sparse import SparseTensor

from torch_sparse import SparseTensor


class SimpleProgressBar:
    """
    Minimal console progress bar to display optimization progress.
    """

    def __init__(
        self,
        max_value: int,
        length: int = 20,
        symbol: str = "=",
        silent_mode: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        max_value : int
            Total number of epochs/steps.
        length : int, optional
            Number of character slots in the bar, by default 20.
        symbol : str, optional
            Marker symbol, by default "=".
        silent_mode : bool, optional
            If True, suppress output entirely.
        """
        self.symbol = symbol
        self.mx = max_value
        self.len = length
        self.delta = self.mx / self.len
        self.ndigits = len(str(self.mx))

        print("\r\n")

        self.call_func = self._silent if silent_mode else self._verbose

    def _verbose(self, epoch: int, value: float) -> None:
        """
        Render the current progress line.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        value : float
            Value to display (e.g., current loss).
        """
        progress = self.symbol * int(epoch / self.delta)
        print(
            f"\rEpoch : {epoch + 1:<{self.ndigits}}/{self.mx:<{self.ndigits}}"
            f" | Loss : {value:9E}"
            f" | \x1b[1;37m["
            f" \x1b[0;36m{progress:<{self.len}}"
            f"\x1b[1;37m]"
            f" \x1b[0m",
            end="",
        )

    def _silent(self, *args, **kwargs) -> NoReturn:
        """Do nothing (silent mode)."""
        pass

    def __call__(self, epoch: int, value: float) -> NoReturn:
        """Update the progress bar."""
        self.call_func(epoch, value)


def construct_interaction(position: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """
    Build a symmetric k-NN adjacency matrix using dense distance computation.
    """
    distance_matrix = cdist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    interaction = np.zeros((n_spot, n_spot))
    for i in range(n_spot):
        nearest = distance_matrix[i].argsort()[1 : n_neighbors + 1]
        interaction[i, nearest] = 1

    adj = interaction + interaction.T
    adj = np.where(adj > 1, 1, adj)
    return adj


def construct_interaction_KNN(position: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """
    Build a symmetric k-NN adjacency matrix via sklearn's NearestNeighbors.
    """
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)

    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()

    interaction = np.zeros((n_spot, n_spot))
    interaction[x, y] = 1

    adj = interaction + interaction.T
    adj = np.where(adj > 1, 1, adj)
    return adj


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalize adjacency matrix D^{-1/2} A D^{-1/2}.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj: np.ndarray) -> np.ndarray:
    """
    Normalize adjacency and add self-loops for vanilla GCN preprocessing.
    """
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> SparseTensor:
    """
    Convert a SciPy sparse matrix into a torch_sparse SparseTensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=shape)


def preprocess_adj_sparse(adj: np.ndarray) -> SparseTensor:
    """
    Normalize adjacency with self-loops and return SparseTensor format.
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def build_adjacency_matrix(cell_locations: np.ndarray, n_neighbo: int = 10) -> coo_matrix:
    """
    Construct a sparse k-NN adjacency using a KDTree.
    """
    kdtree = KDTree(cell_locations)
    _, indices = kdtree.query(cell_locations, k=n_neighbo + 1)
    indices = indices[:, 1:]

    n_spot = cell_locations.shape[0]
    row_indices = np.repeat(np.arange(n_spot), n_neighbo)
    col_indices = indices.flatten()
    data = np.ones_like(row_indices, dtype=np.float32)

    return coo_matrix((data, (row_indices, col_indices)), shape=(n_spot, n_spot))


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer labels to one-hot encodings.
    """
    eye = torch.eye(num_classes)
    return eye[labels]


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy for logits vs. ground-truth labels.
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)