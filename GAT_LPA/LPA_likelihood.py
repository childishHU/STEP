import torch
import torch.nn.functional as F
from .utils_GAT import sparse_mx_to_torch_sparse_tensor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.sparse as sp
scaler = StandardScaler()


def cosine_similarity_matrix(x):
    x_normalized = F.normalize(x, p=2, dim=1)
    similarity_matrix = torch.matmul(x_normalized, x_normalized.t())
    return similarity_matrix


def LPA_likelihood(epochs, features, labels, nclass, cell_locations, idx_train, n_neighbo=10, likelihood=True):

    cell_locations = torch.from_numpy(np.array(cell_locations)).to(torch.float32)
    output = torch.cdist(cell_locations, cell_locations)
    _ , indices = torch.topk(output, n_neighbo + 1, largest=False)
    x = indices[:, 0].repeat_interleave(n_neighbo)
    y = indices[:, 1:].flatten()
    n_spot = cell_locations.shape[0]
    interaction = torch.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    adj = interaction
    adj = adj + adj.T
    adj = torch.where(adj>1, 1, adj)
    df_j = indices[:, 1:]
    adj = sp.coo_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    y = torch.eye(nclass)
    labels_lpa = y[labels]
    res_labels_lpa = labels_lpa.clone()
    for i in tqdm(range(epochs)):
        new_labels_lpa = adj @ res_labels_lpa
        new_labels_lpa = new_labels_lpa / new_labels_lpa.sum(axis=1)[:,None]
        res_labels_lpa = new_labels_lpa.clone()
        res_labels_lpa[idx_train] = labels_lpa[idx_train]
    
    res = torch.argmax(res_labels_lpa, dim=1)
    res_final = res.clone()
    if likelihood:
        features_ = torch.from_numpy(scaler.fit_transform(features))
        cellWeight = torch.exp(cosine_similarity_matrix(features_))

        for idx in list(set(range(len(res))) - set(idx_train)):
            j_vector = df_j[idx]
            score_vec = torch.zeros(nclass)
            if j_vector.shape[0] != 0:
                for ctype in range(nclass):
                        prior = ((res[j_vector] != ctype) * cellWeight[idx, j_vector]).sum() * n_neighbo / j_vector.shape[0]
                        #prior += ((res[j_vector] == ctype) * (1 + 1 / cellWeight[idx, j_vector])).sum() * n_neighbo / j_vector.shape[0]
                        score_vec[ctype] = prior
                res_final[idx] = torch.argmin(score_vec)
    
    return res_final.detach().cpu().numpy()
    
    
