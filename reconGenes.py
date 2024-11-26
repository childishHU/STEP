import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc

def SearchInType(InitProp, cell_locations, x_beta, adata_st, nu=50,spatial_balance=True):
    device = torch.device(InitProp['config']['device'])
    
    gene_list = InitProp['internal_vars']['gene_list_reg']
    signature_matrix = InitProp['cell_type_info']['info']['cell_type_means'].loc[gene_list, cell_locations['discrete_label_ct']]
    signature_matrix = torch.tensor(signature_matrix.values, dtype=torch.float32).to(device) * torch.tensor(x_beta.T, dtype=torch.float32).to(device)

    # Normalize reference data
    ref_counts = InitProp['reference']['counts']
    ref_org_all = torch.tensor(ref_counts.T.values, dtype=torch.float32).to(device)
    ref_norm_all = ref_org_all / ref_org_all.sum(dim=1, keepdim=True)

    ref_org = torch.tensor(ref_counts.loc[gene_list].T.values, dtype=torch.float32).to(device)
    ref_norm = ref_org / ref_org.sum(dim=1, keepdim=True)

    # Create reference dictionaries by cell type
    ref_cell_type = InitProp['reference']['cell_types'].values.reshape(-1)
    unique_cell_types = np.unique(ref_cell_type)
    ref_norm_dict_all = {ct: ref_norm_all[ref_cell_type == ct] for ct in unique_cell_types}
    ref_norm_dict = {ct: ref_norm[ref_cell_type == ct] for ct in unique_cell_types}
    ref_org_dict_all = {ct: ref_org_all[ref_cell_type == ct] for ct in unique_cell_types}


    # Fit NearestNeighbors models
    nbs_dict = {ct: NearestNeighbors(n_neighbors=min(nu, len(ref_norm_dict[ct])), algorithm='auto', metric='cosine').fit(ref_norm_dict[ct].detach().cpu().numpy()) for ct in unique_cell_types}

    new_signature_matrix = torch.zeros((ref_org_all.shape[1], len(cell_locations)), dtype=torch.float32).to(device)

    # Update signature matrix
    cell_type = cell_locations['discrete_label_ct']
    for label in np.unique(cell_type):
        idx = np.where(cell_type == label)[0]
        sig_matrix_label = signature_matrix[:, idx].detach().cpu().numpy().T
        distances, indices = nbs_dict[label].kneighbors(sig_matrix_label)

        distances = torch.tensor(distances, dtype=torch.float32).to(device)
        indices = torch.tensor(indices, dtype=torch.int64).to(device)

        valid_mask = distances < 1
        valid_distances = torch.where(valid_mask, distances, torch.zeros_like(distances))
        weights = 1 - valid_distances / valid_distances.sum(dim=1, keepdim=True)
        weights /= (valid_mask.sum(dim=1, keepdim=True) - 1)

        for j in range(len(idx)):
            if valid_mask[j].sum().item() > 1:
                selected_indices = indices[j][valid_mask[j]]
                weighted_sum = torch.matmul(weights[j, valid_mask[j]], ref_org_dict_all[label][selected_indices])
                new_signature_matrix[:, idx[j]] = weighted_sum
            else:
                new_signature_matrix[:, idx[j]] = ref_org_dict_all[label][indices[j, 0]]
    
    del ref_org_all, ref_norm_all, ref_org, ref_norm, ref_norm_dict_all, ref_norm_dict, ref_org_dict_all, signature_matrix
    common_gene = np.intersect1d(adata_st.var.index.values, InitProp['reference']['counts'].index.values)
    adata_st = adata_st[:, common_gene]
    #sc.pp.normalize_total(adata_st, target_sum=1)
    gene_to_index = pd.DataFrame(np.arange(ref_counts.shape[0]),index=ref_counts.index)
    common_gene_index = gene_to_index.loc[common_gene].values.reshape(-1)
    new_signature_matrix = new_signature_matrix[common_gene_index, :]
    temp = torch.sum(new_signature_matrix) 
    if spatial_balance:

        minmax = MinMaxScaler()
        cell = minmax.fit_transform(cell_locations[['x','y']].values)
        spot = minmax.transform(adata_st.obsm['spatial'])
        ED = cdist(cell, spot)

        isigma = 0.1
        kernel_mat = np.exp(-ED ** 2 / (2 * isigma ** 2))    
        k = 5

        results = np.zeros_like(kernel_mat)
        for i in range(kernel_mat.shape[0]):
            indices = np.argpartition(kernel_mat[i], -k)[-k:]
            results[i, indices] = kernel_mat[i, indices]
            
        try:
            stx = torch.tensor(adata_st.X.toarray(), dtype=torch.float32).to(device)
        except:
            stx = torch.tensor(adata_st.X, dtype=torch.float32).to(device)

        results = torch.tensor(results, dtype=torch.float32).to(device)
        results = results @ stx
        results = results / torch.sum(results)

        alpha = 0.98
        new_signature_matrix = new_signature_matrix * results.T 
        new_signature_matrix = new_signature_matrix / torch.sum(new_signature_matrix)
        new_signature_matrix = (alpha * new_signature_matrix + (1 - alpha) * results.T) *  temp 
        

        del results, stx 

    return pd.DataFrame(new_signature_matrix.detach().cpu().numpy(), index=common_gene, columns=cell_locations.index)

def SearchImage(RNA_data, RNA_meta, new_sc, new_st, new_genes, cell_locations, class_name, subClass_name):

    class_label = np.unique(RNA_meta[class_name])
    transfer = dict(zip(RNA_meta[subClass_name], RNA_meta[class_name]))
    ref_cell_type = RNA_meta[class_name].values.reshape(-1)
    Y_train = RNA_data.T[new_genes]
    Y = {ct: Y_train.iloc[ref_cell_type == ct] for ct in class_label}
    ref = {ct: new_sc.iloc[ref_cell_type == ct] for ct in class_label}
    Imp_Gene = np.zeros((new_st.shape[0], len(new_genes)))

    nbs_dict = {ct: NearestNeighbors(n_neighbors=min(50, len(ref[ct])), algorithm='auto', metric='cosine').fit(ref[ct]) for ct in class_label}
    
    pre_cell_type = cell_locations['discrete_label_ct'].values.reshape(-1)
    pre_cell_type = np.array([transfer[c]  for c in pre_cell_type])
    for label in np.unique(pre_cell_type):
        idx = np.where(pre_cell_type == label)[0]
        new_st_label = new_st.iloc[idx, :]
        distances, indices = nbs_dict[label].kneighbors(new_st_label)

        for j in range(len(idx)):
            weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]))
            weights = weights/(len(weights)-1)
            Imp_Gene[idx[j]]= np.dot(weights,Y[label].iloc[indices[j,:][distances[j,:] < 1]])

    Imp_Gene[np.isnan(Imp_Gene)] = 0
    return Imp_Gene


def SearchTransfer(feature_from, feature_to, signature_matrix_from):
    signature_matrix_from_tensor = torch.tensor(signature_matrix_from.values, dtype=torch.float32)

    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='cosine').fit(feature_from)
    distances, indices = nbrs.kneighbors(feature_to)

    distances_tensor = torch.tensor(distances, dtype=torch.float32)
    indices_tensor = torch.tensor(indices, dtype=torch.int64)

    signature_matrix_to_tensor = torch.zeros((feature_to.shape[0], signature_matrix_from.shape[1]), dtype=torch.float32)

    for j in range(feature_to.shape[0]):
        dist_j = distances_tensor[j]
        ind_j = indices_tensor[j]

        mask = dist_j < 1.0
        filtered_distances = dist_j[mask]
        filtered_indices = ind_j[mask]

        weights = 1 - filtered_distances / torch.sum(filtered_distances)
        weights /= (len(weights) - 1)

        selected_signatures = signature_matrix_from_tensor[filtered_indices]
        weighted_sum = torch.matmul(weights, selected_signatures)

        signature_matrix_to_tensor[j] = weighted_sum

    signature_matrix_to = pd.DataFrame(signature_matrix_to_tensor.numpy(), index=feature_to.index, columns=signature_matrix_from.columns)

    return signature_matrix_to

