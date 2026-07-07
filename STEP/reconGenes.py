import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
import scipy.stats as st


def SearchInType(InitProp, cell_locations, x_beta, adata_st,
                 nu=50, spatial_balance=True, process_log=None):
    """
    Refine a cell-type signature matrix by finding nearest reference cells
    and optionally applying a spatial balancing kernel.

    Parameters
    ----------
    InitProp : dict
        Initialization package containing configuration, reference data, etc.
    cell_locations : pandas.DataFrame
        Per-cell table with discrete labels and coordinates.
    x_beta : numpy.ndarray
        Scaling factors (one per signature component).
    adata_st : AnnData
        Spatial transcriptomics object.
    nu : int, optional
        Number of neighbors used by the k-NN search (default: 50).
    spatial_balance : bool, optional
        If True, blend signatures with spot expression using a spatial kernel.
    process_log : str or None
        File path used to report progress; guarded by FileLock.
    """
    device = torch.device(InitProp['config']['device'])

    # ------------------------------------------------------------------
    # Build initial signature matrix (gene x cell) limited to the target genes.
    # ------------------------------------------------------------------
    gene_list = InitProp['internal_vars']['gene_list_reg']
    sig_df = InitProp['cell_type_info']['info']['cell_type_means'].loc[
        gene_list, cell_locations['discrete_label_ct']
    ]
    signature_matrix = (
        torch.tensor(sig_df.values, dtype=torch.float32, device=device) *
        torch.tensor(x_beta.T, dtype=torch.float32, device=device)
    )

    # ------------------------------------------------------------------
    # Prepare reference (scRNA-seq) matrices in original and normalized form.
    # ------------------------------------------------------------------
    ref_counts = InitProp['reference']['counts']

    ref_org_all = torch.tensor(ref_counts.T.values, dtype=torch.float32, device=device)
    row_sums_all = ref_org_all.sum(dim=1, keepdim=True)
    row_sums_all[row_sums_all == 0] = 1.0
    ref_norm_all = torch.nan_to_num(ref_org_all / row_sums_all)

    ref_org = torch.tensor(ref_counts.loc[gene_list].T.values, dtype=torch.float32, device=device)
    row_sums = ref_org.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    ref_norm = torch.nan_to_num(ref_org / row_sums)

    # Partition reference cells by cell type for subsequent lookups.
    ref_cell_type = InitProp['reference']['cell_types'].values.reshape(-1)
    unique_cell_types = np.unique(ref_cell_type)
    ref_norm_dict_all = {ct: ref_norm_all[ref_cell_type == ct] for ct in unique_cell_types}
    ref_norm_dict = {ct: ref_norm[ref_cell_type == ct] for ct in unique_cell_types}
    ref_org_dict_all = {ct: ref_org_all[ref_cell_type == ct] for ct in unique_cell_types}

    # ------------------------------------------------------------------
    # Fit k-NN models (cosine similarity) per cell type on normalized refs.
    # ------------------------------------------------------------------
    nbs_dict = {
        ct: NearestNeighbors(
            n_neighbors=min(nu, len(ref_norm_dict[ct])),
            algorithm='auto',
            metric='cosine'
        ).fit(ref_norm_dict[ct].detach().cpu().numpy())
        for ct in unique_cell_types
    }

    # Placeholder for the refined signature matrix (all reference genes x cells)
    new_signature_matrix = torch.zeros(
        (ref_org_all.shape[1], len(cell_locations)),
        dtype=torch.float32,
        device=device
    )

    # ------------------------------------------------------------------
    # For each predicted cell type, refine signatures via weighted neighbors.
    # ------------------------------------------------------------------
    cell_type = cell_locations['discrete_label_ct']
    unique_labels = np.unique(cell_type)

    for ii, label in enumerate(unique_labels):
        if process_log is not None:
            with FileLock(process_log + '.lock'):
                with open(process_log, 'w') as f:
                    f.write(str(int(20 + (ii + 1) / len(unique_labels) * 60)))

        idx = np.where(cell_type == label)[0]
        sig_matrix_label = signature_matrix[:, idx].detach().cpu().numpy().T
        distances, indices = nbs_dict[label].kneighbors(sig_matrix_label)

        distances = torch.tensor(distances, dtype=torch.float32, device=device)
        indices = torch.tensor(indices, dtype=torch.int64, device=device)

        valid_mask = distances < 1
        valid_distances = torch.where(valid_mask, distances, torch.zeros_like(distances))
        weights = 1 - valid_distances / valid_distances.sum(dim=1, keepdim=True)
        weights /= (valid_mask.sum(dim=1, keepdim=True) - 1)

        for j, cell_idx in enumerate(idx):
            if valid_mask[j].sum().item() > 1:
                selected = indices[j][valid_mask[j]]
                weighted_sum = torch.matmul(weights[j, valid_mask[j]], ref_org_dict_all[label][selected])
                new_signature_matrix[:, cell_idx] = weighted_sum
            else:
                new_signature_matrix[:, cell_idx] = ref_org_dict_all[label][indices[j, 0]]

    # ------------------------------------------------------------------
    # Restrict to genes shared with the spatial dataset and rescale.
    # ------------------------------------------------------------------
    del ref_org_all, ref_norm_all, ref_org, ref_norm
    del ref_norm_dict_all, ref_norm_dict, ref_org_dict_all, signature_matrix

    common_gene = np.intersect1d(adata_st.var.index.values, ref_counts.index.values)
    adata_st = adata_st[:, common_gene]
    gene_to_index = pd.DataFrame(np.arange(ref_counts.shape[0]), index=ref_counts.index)
    common_gene_index = gene_to_index.loc[common_gene].values.reshape(-1)
    new_signature_matrix = new_signature_matrix[common_gene_index, :]
    total_sum = torch.sum(new_signature_matrix)

    if process_log is not None:
        with FileLock(process_log + '.lock'):
            with open(process_log, 'w') as f:
                f.write('80')

    # ------------------------------------------------------------------
    # Optional spatial balancing using a Gaussian kernel over coordinates.
    # ------------------------------------------------------------------
    if spatial_balance:
        scaler = MinMaxScaler()
        cell = scaler.fit_transform(cell_locations[['x', 'y']].values)
        spot = scaler.transform(adata_st.obsm['spatial'])
        euclid_dist = cdist(cell, spot)

        isigma = 0.1
        kernel_mat = np.exp(-euclid_dist ** 2 / (2 * isigma ** 2))

        k = 5
        results = np.zeros_like(kernel_mat)
        for i in range(kernel_mat.shape[0]):
            top_k_idx = np.argpartition(kernel_mat[i], -k)[-k:]
            results[i, top_k_idx] = kernel_mat[i, top_k_idx]

        try:
            stx = torch.tensor(adata_st.X.toarray(), dtype=torch.float32, device=device)
        except AttributeError:
            stx = torch.tensor(adata_st.X, dtype=torch.float32, device=device)

        results = torch.tensor(results, dtype=torch.float32, device=device)
        results = results @ stx
        results = results / torch.sum(results)

        alpha = 0.95
        new_signature_matrix = new_signature_matrix * results.T
        new_signature_matrix = new_signature_matrix / torch.sum(new_signature_matrix)
        new_signature_matrix = (alpha * new_signature_matrix + (1 - alpha) * results.T) * total_sum

        del results, stx

    # Return a pandas DataFrame for downstream compatibility
    return pd.DataFrame(
        new_signature_matrix.detach().cpu().numpy(),
        index=common_gene,
        columns=cell_locations.index
    )


def SearchImage(RNA_data, RNA_meta, new_sc, new_st, new_genes,
                cell_locations, class_name, subClass_name, nu=50, process_log=None):
    """
    Transfer signatures from reference RNA data to spatial cells via k-NN search.

    Parameters
    ----------
    RNA_data : pandas.DataFrame
        Reference gene expression matrix (cells x genes).
    RNA_meta : pandas.DataFrame
        Metadata containing class and subclass annotations.
    new_sc : pandas.DataFrame
        Feature space used for neighbor search (reference).
    new_st : pandas.DataFrame
        Feature space for spatial cells (query).
    new_genes : list
        Genes shared between reference and spatial data.
    cell_locations : pandas.DataFrame
        Contains the predicted discrete cell types (`discrete_label_ct`).
    class_name : str
        Column in `RNA_meta` representing coarse classes (e.g., major cell types).
    subClass_name : str
        Column representing fine subclasses used for mapping.
    nu : int, optional
        Number of neighbors for k-NN search (default: 50).
    process_log : str or None
        Optional path for progress logging.
    """
    # Map subclass labels to class labels for consistency
    transfer_map = dict(zip(RNA_meta[subClass_name], RNA_meta[class_name]))
    pre_cell_type = cell_locations['discrete_label_ct'].values.reshape(-1)
    pre_cell_type = np.array([transfer_map.get(ct, ct) for ct in pre_cell_type])

    class_label = np.unique(pre_cell_type)
    ref_cell_type = RNA_meta[class_name].values.reshape(-1)

    # Create class-specific reference data structures
    Y_train = RNA_data.T[new_genes]
    Y = {ct: Y_train.iloc[ref_cell_type == ct] if (ref_cell_type == ct).any() else Y_train
         for ct in class_label}
    ref = {ct: new_sc.iloc[ref_cell_type == ct] if (ref_cell_type == ct).any() else new_sc
           for ct in class_label}

    Imp_Gene = np.zeros((new_st.shape[0], len(new_genes)))

    nbs_dict = {
        ct: NearestNeighbors(
            n_neighbors=min(nu, len(ref[ct])),
            algorithm='auto',
            metric='cosine'
        ).fit(ref[ct])
        for ct in class_label
    }

    unique_types = np.unique(pre_cell_type)
    for ii, label in enumerate(unique_types):
        if process_log is not None:
            with FileLock(process_log + '.lock'):
                with open(process_log, 'w') as f:
                    f.write(str(int(20 + (ii + 1) / len(unique_types) * 80)))

        idx = np.where(pre_cell_type == label)[0]
        new_st_label = new_st.iloc[idx, :]
        distances, indices = nbs_dict[label].kneighbors(new_st_label)

        for j, global_idx in enumerate(idx):
            mask = distances[j, :] < 1
            if mask.sum() <= 1:
                continue  # Avoid division by zero when only one neighbor passes the threshold

            valid_distances = distances[j, mask]
            weights = 1 - valid_distances / np.sum(valid_distances)
            weights /= (len(weights) - 1)

            Imp_Gene[global_idx] = np.dot(weights, Y[label].iloc[indices[j, mask]])

    Imp_Gene[np.isnan(Imp_Gene)] = 0
    return Imp_Gene


def SearchTransfer(feature_from, feature_to, signature_matrix_from):
    """
    Transfer a signature matrix to a new feature space using nearest neighbors.
    """
    signature_matrix_from_tensor = torch.tensor(signature_matrix_from.values, dtype=torch.float32)

    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='cosine').fit(feature_from)
    distances, indices = nbrs.kneighbors(feature_to)

    distances_tensor = torch.tensor(distances, dtype=torch.float32)
    indices_tensor = torch.tensor(indices, dtype=torch.int64)

    signature_matrix_to_tensor = torch.zeros(
        (feature_to.shape[0], signature_matrix_from.shape[1]),
        dtype=torch.float32
    )

    for j in range(feature_to.shape[0]):
        mask = distances_tensor[j] < 1.0
        if mask.sum() <= 1:
            signature_matrix_to_tensor[j] = signature_matrix_from_tensor[indices_tensor[j, 0]]
            continue

        filtered_distances = distances_tensor[j][mask]
        filtered_indices = indices_tensor[j][mask]

        weights = 1 - filtered_distances / torch.sum(filtered_distances)
        weights /= (len(weights) - 1)

        selected_signatures = signature_matrix_from_tensor[filtered_indices]
        weighted_sum = torch.matmul(weights, selected_signatures)

        signature_matrix_to_tensor[j] = weighted_sum

    signature_matrix_to = pd.DataFrame(
        signature_matrix_to_tensor.numpy(),
        index=feature_to.index,
        columns=signature_matrix_from.columns
    )
    return signature_matrix_to


def searchProtein(protein_, gene_, protein_name, nu=50, use_zscore=True):
    """
    Infer protein abundances for gene-expression cells by referencing matched protein data.

    Parameters
    ----------
    protein_ : AnnData
        CITE-seq (or similar) data containing protein measurements.
    gene_ : AnnData
        Gene expression data aligned to the same proteins.
    protein_name : list-like
        Names of protein markers to transfer.
    nu : int, optional
        Number of neighbors for k-NN search (default: 50).
    use_zscore : bool, optional
        Whether to z-score both modalities before matching.
    """
    common_gene = np.intersect1d(gene_.var_names, protein_.var_names)

    protein = protein_[:, common_gene].to_df()
    gene = gene_[:, common_gene].to_df()

    if use_zscore:
        gene = pd.DataFrame(st.zscore(gene, axis=0), index=gene.index, columns=gene.columns).fillna(0)
        protein = pd.DataFrame(st.zscore(protein, axis=0), index=protein.index, columns=protein.columns).fillna(0)

    Y_train = protein_.obs[protein_name]

    nbrs = NearestNeighbors(n_neighbors=nu, algorithm='auto', metric='cosine').fit(protein)
    distances, indices = nbrs.kneighbors(gene)

    cellProtein = pd.DataFrame(
        np.zeros((gene.shape[0], len(protein_name))),
        columns=protein_name,
        index=gene.index
    )

    for j in range(gene.shape[0]):
        mask = distances[j, :] < 1
        if mask.sum() <= 1:
            continue  # Not enough neighbors for weighting

        valid_dist = distances[j, mask]
        weights = 1 - valid_dist / np.sum(valid_dist)
        weights /= (len(weights) - 1)
        cellProtein.iloc[j, :] = np.dot(weights, Y_train.iloc[indices[j, mask]])

    return cellProtein