import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .utils_GAT import sparse_mx_to_torch_sparse_tensor

scaler = StandardScaler()


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine-similarity matrix for a batch of feature vectors.

    Parameters
    ----------
    x : torch.Tensor
        Input matrix of shape (N, D) where rows are feature vectors.

    Returns
    -------
    torch.Tensor
        Cosine similarity matrix of shape (N, N).
    """
    x_normalized = F.normalize(x, p=2, dim=1)
    similarity_matrix = torch.matmul(x_normalized, x_normalized.t())
    return similarity_matrix


def LPA_likelihood(
    epochs: int,
    features: np.ndarray,
    labels: np.ndarray,
    nclass: int,
    cell_locations: np.ndarray,
    idx_train: np.ndarray,
    n_neighbo: int = 10,
    likelihood: bool = True,
) -> np.ndarray:
    """
    Run label propagation on a k-NN graph optionally refined by a likelihood heuristic.

    Parameters
    ----------
    epochs : int
        Number of propagation iterations.
    features : np.ndarray
        Feature matrix used for cosine-similarity weighting (N, D).
    labels : np.ndarray
        Initial integer labels for nodes.
    nclass : int
        Number of classes.
    cell_locations : np.ndarray
        Coordinates for building the k-NN graph (N, dim).
    idx_train : np.ndarray
        Indices of labeled nodes to keep fixed during propagation.
    n_neighbo : int, optional
        Number of nearest neighbors for graph construction, by default 10.
    likelihood : bool, optional
        Whether to apply the likelihood-based refinement, by default True.

    Returns
    -------
    np.ndarray
        Final predicted class labels per node.
    """
    # Build k-NN graph from spatial coordinates
    cell_locations = torch.from_numpy(np.array(cell_locations)).to(torch.float32)
    dist = torch.cdist(cell_locations, cell_locations)
    _, indices = torch.topk(dist, n_neighbo + 1, largest=False)

    x = indices[:, 0].repeat_interleave(n_neighbo)
    y = indices[:, 1:].flatten()
    n_spot = cell_locations.shape[0]

    interaction = torch.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    adj = interaction + interaction.T
    adj = torch.where(adj > 1, 1, adj)  # Remove duplicate edges
    df_j = indices[:, 1:]

    # Convert adjacency to sparse tensor for efficient multiplication
    adj = sp.coo_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Initialize label distributions (one-hot for labeled nodes)
    eye = torch.eye(nclass)
    labels_lpa = eye[labels]
    res_labels_lpa = labels_lpa.clone()

    # Run vanilla label propagation
    for _ in tqdm(range(epochs)):
        new_labels_lpa = adj @ res_labels_lpa
        new_labels_lpa = new_labels_lpa / (new_labels_lpa.sum(axis=1, keepdim=True) + 1e-12)
        res_labels_lpa = new_labels_lpa.clone()
        res_labels_lpa[idx_train] = labels_lpa[idx_train]

    res = torch.argmax(res_labels_lpa, dim=1)
    res_final = res.clone()

    if likelihood:
        # Likelihood refinement using cosine-similarity-based weights
        features_ = torch.from_numpy(scaler.fit_transform(features)).to(torch.float32)
        cell_weight = torch.exp(cosine_similarity_matrix(features_))

        unlabeled = list(set(range(len(res))) - set(idx_train))
        for idx in unlabeled:
            neighbors = df_j[idx]
            score_vec = torch.zeros(nclass)

            if neighbors.shape[0] != 0:
                for ctype in range(nclass):
                    mismatch = (res[neighbors] != ctype).float()
                    weighted = mismatch * cell_weight[idx, neighbors]
                    prior = weighted.sum() * n_neighbo / neighbors.shape[0]
                    score_vec[ctype] = prior

                res_final[idx] = torch.argmin(score_vec)

    return res_final.detach().cpu().numpy()