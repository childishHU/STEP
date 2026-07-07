import sys
import pandas as pd
import numpy as np
import os
from scipy.sparse import identity, diags, csr_matrix
from scipy.spatial import distance_matrix, cKDTree
from scipy.spatial.distance import cdist, pdist, squareform
import itertools
from .utils_MH import *
from scipy.sparse import diags
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_type = 'float32'

import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def find_neighbors_fs(position, k=10):
    """
    Fast KNN neighbor search (fixed k) using cKDTree.
    """
    tree = cKDTree(position, leafsize=40)
    _, indices = tree.query(position, k=k + 1)
    return [ind[1:] for ind in indices]


def find_neighbors(position, q=0.004, p=1):
    """
    Radius-based neighbor search selecting radius from q-th quantile.
    """
    pairwise = distance_matrix(position, position, p=p)
    radius = np.quantile(pairwise[pairwise != 0], q)
    mask = (pairwise <= radius) & (pairwise > 0)
    return [np.where(mask[i])[0] for i in range(mask.shape[0])]


def KeepOrderUnique(seq):
    """
    Preserve order while removing duplicates.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def compute_neighbor_cosine_similarity(X, neighbors):
    """
    Build sparse cosine-similarity matrix restricted to neighbor pairs.
    """
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    n, _ = X_np.shape
    norms = np.linalg.norm(X_np, axis=1)
    safe_norms = np.where(norms > 0, norms, 1.0)

    row_idx_list, col_idx_list = [], []
    for i, nbrs in enumerate(neighbors):
        if len(nbrs) == 0:
            continue
        row_idx_list.append(np.full(len(nbrs), i, dtype=np.int64))
        col_idx_list.append(np.asarray(nbrs, dtype=np.int64))

    if not row_idx_list:
        return csr_matrix((n, n))

    row_idx = np.concatenate(row_idx_list)
    col_idx = np.concatenate(col_idx_list)
    xi = X_np[row_idx]
    xj = X_np[col_idx]
    dot_products = np.einsum('ij,ij->i', xi, xj)
    cosine_sim = dot_products / (safe_norms[row_idx] * safe_norms[col_idx] + 1e-10)

    return csr_matrix((cosine_sim, (row_idx, col_idx)), shape=(n, n))


@ray.remote
def UpdateCellLabel_Greedy(args, xbeta):
    """
    Greedy local re-labeling of cells within a single spot (Ray remote task).
    """
    Y, index, pos_celltype, nUMI, alpha, cell_num_total, nu, Spot_index = args

    # Single-cell case: enumerate candidate types and pick best likelihood+prior.
    if index.shape[0] == 1:
        init_labels = init['k']  # read-only here; no full-array copy needed
        score_vec = np.zeros(pos_celltype.shape[0])
        choose_from = pos_celltype
        scaler = nUMI * alpha / cell_num_total
        j_vector = df_j[index[0]]

        for k in range(choose_from.shape[0]):
            j = choose_from[k]
            mu_hat = (
                signature_matrix
                .loc[:, cell_type_names[j]]
                .values.squeeze() *
                xbeta[:, index[0]].squeeze()
            )
            prediction = mu_hat * scaler if P is None else mu_hat * scaler * P[Spot_index, index[0]]
            likelihood = calc_log_l_vec(prediction.reshape(-1), Y, likelihood_vars=likelihood_vars)
            score_vec[k] = likelihood

            if j_vector.shape[0] != 0:
                if P is None:
                    prior = (init_labels[j_vector] != j).sum() * nu / j_vector.shape[0]
                else:
                    if isinstance(cellWeight, (int, float)):
                        prior = (
                            (init_labels[j_vector] != j) * cellWeight
                        ).sum() * nu / j_vector.shape[0]
                    else:
                        prior = (
                            (init_labels[j_vector] != j) *
                            cellWeight[index[0], j_vector].toarray()
                        ).sum() * nu / j_vector.shape[0]
                score_vec[k] += prior

        return index, np.array([choose_from[np.argmin(score_vec)]])

    # Multi-cell case (pair swapping search).
    else:
        init_k = init['k']  # operate directly on the worker's global array, no copy
        scaler = alpha * nUMI / cell_num_total
        for h in np.random.choice(np.arange(len(com[index.shape[0] - 2])), len(com[index.shape[0] - 2]), replace=False):
            cells_index_small = com[index.shape[0] - 2][h]
            cell_index = index[np.array(cells_index_small)]
            other_index = init_k[index[~np.isin(index, cell_index)]]

            temp_weight = np.zeros((1, cell_types))
            if other_index.shape[0] == 0:
                other_mu = 0
            else:
                p_temp = np.array([P[Spot_index, i] for i in index[~np.isin(index, cell_index)]])
                other_mu = (
                    signature_matrix.loc[:, [cell_type_names[_] for _ in other_index]].values *
                    xbeta[:, index[~np.isin(index, cell_index)]] *
                    p_temp[None, :]
                ).sum(-1)
                for j in range(other_index.shape[0]):
                    temp_weight[0, other_index[j]] += p_temp[j]

            ct_all = list(itertools.product(pos_celltype, repeat=2))
            j_vector1 = df_j[cell_index[0]]
            j_vector2 = df_j[cell_index[1]]
            score_vec = np.zeros(len(ct_all))
            p_temp = np.array([P[Spot_index, i] for i in cell_index])

            for p in np.random.choice(np.arange(len(ct_all)), len(ct_all), replace=False):
                ct = ct_all[p]
                mu_hat = (
                    signature_matrix.loc[:, [cell_type_names[_] for _ in ct]].values *
                    xbeta[:, cell_index] *
                    p_temp[None, :]
                ).sum(-1)
                mu_hat += other_mu
                prediction = mu_hat * scaler
                likelihood = calc_log_l_vec(prediction.reshape(-1), Y, likelihood_vars=likelihood_vars)

                if (j_vector1.shape[0] == 0) and (j_vector2.shape[0] == 0):
                    score_vec[p] = likelihood
                elif (j_vector1.shape[0] == 0) and (j_vector2.shape[0] != 0):
                    if isinstance(cellWeight, (int, float)):
                        prior2 = (
                        (init_k[j_vector2] != ct[1]) *
                            cellWeight
                        ).sum() * nu / j_vector2.shape[0]
                    else:
                        prior2 = (
                            (init_k[j_vector2] != ct[1]) *
                            cellWeight[cell_index[1], j_vector2].toarray()
                        ).sum() * nu / j_vector2.shape[0]
                    score_vec[p] = likelihood + prior2
                elif (j_vector1.shape[0] != 0) and (j_vector2.shape[0] == 0):
                    if isinstance(cellWeight, (int, float)):
                        prior1 = (
                            (init_k[j_vector1] != ct[0]) *
                            cellWeight
                        ).sum() * nu / j_vector1.shape[0]
                    else:
                        prior1 = (
                            (init_k[j_vector1] != ct[0]) *
                            cellWeight[cell_index[0], j_vector1].toarray()
                        ).sum() * nu / j_vector1.shape[0]
                    score_vec[p] = likelihood + prior1
                else:
                    Ni = 0
                    for c in pos_celltype:
                        Ni += SmoothPrior(c, ct[1], init_k, j_vector1, cell_index[1], nu, cellWeight) / \
                              SmoothPrior(ct[1], c, init_k, j_vector2, cell_index[0], nu, cellWeight)
                    logNi = np.log(Ni)
                    prior = SmoothPrior(ct[0], ct[1], init_k, j_vector1, cell_index[1], nu, cellWeight, logp=True) - logNi
                    score_vec[p] = likelihood - prior

            init_k[cell_index] = np.array(ct_all[np.argmin(score_vec)])

    return index, init_k[index]


def SingleCellTypeIdentification(
    InitProp,
    spot_index_name,
    Q_mat_all,
    X_vals_loc,
    nu=0,
    n_epoch=8,
    n_neighbo=10,
    loggings=None,
    hs_ST=False,
    process_log=None
):
    """
    Main iterative procedure for assigning discrete cell labels via greedy + MH refinement.
    """
    global com, cell_type_names, weights, cell_types
    global signature_matrix, init, df_j, P, cellWeight, likelihood_vars

    cell_locations = InitProp['imageInfo']['cell_locations'].copy()

    # Choose neighborhood strategy depending on dataset size / presence of z-axis.
    if 'z' in cell_locations.columns and cell_locations['z'].dtype in [np.float64, np.float32, int]:
        if cell_locations.shape[0] > 30000:
            df_j = find_neighbors_fs(cell_locations[['x', 'y', 'z']].values, k=n_neighbo)
        else:
            df_j = find_neighbors(cell_locations[['x', 'y', 'z']].values, q=n_neighbo / cell_locations.shape[0])
    else:
        if cell_locations.shape[0] > 30000:
            df_j = find_neighbors_fs(cell_locations[['x', 'y']].values, k=n_neighbo)
        else:
            df_j = find_neighbors(cell_locations[['x', 'y']].values, q=n_neighbo / cell_locations.shape[0])

    device = torch.device(InitProp['config']['device'])
    cell_types = InitProp['cell_type_info']['renorm']['n_cell_types']
    sp_index = np.array(KeepOrderUnique(cell_locations[spot_index_name]))
    sp_index_table = pd.DataFrame(np.arange(sp_index.shape[0]), index=sp_index)

    P = None
    if InitProp['imageInfo']['partion'] is not None:
        _p = InitProp['imageInfo']['partion'].loc[sp_index, cell_locations.index]
        if hasattr(_p, 'sparse'):
            P = csr_matrix(_p.sparse.to_coo())
        else:
            P = csr_matrix(_p.values)

    init = {}
    MH = True

    # Determine weights / cell-type names per spot.
    if hs_ST:
        try:
            weights = InitProp['results']['weights'].loc[sp_index].values
            cell_type_names = InitProp['results']['weights'].columns.values
        except Exception:
            weights = InitProp['results'].loc[sp_index].values
            cell_type_names = InitProp['results'].columns.values

        if InitProp['imageInfo']['features'] is None:
            MH = False
    else:
        weights = InitProp['results'].loc[sp_index].values
        cell_type_names = InitProp['results'].columns.values

    # Initialize MCMC structures.
    if MH:
        X = pd.DataFrame(
            scaler.fit_transform(InitProp['imageInfo']['features']),
            index=InitProp['imageInfo']['features'].index,
            columns=InitProp['imageInfo']['features'].columns
        ).loc[cell_locations.index, :]
        cellWeight = compute_neighbor_cosine_similarity(X, df_j)
        init['Beta'] = pd.DataFrame(
            np.zeros((X.shape[1], len(InitProp['internal_vars']['gene_list_reg']))),
            index=X.columns,
            columns=InitProp['internal_vars']['gene_list_reg']
        ).astype(data_type)
        init['Gamma'] = pd.DataFrame(
            np.random.choice([0, 1], size=(X.shape[1], len(InitProp['internal_vars']['gene_list_reg'])), replace=True),
            index=X.columns,
            columns=InitProp['internal_vars']['gene_list_reg']
        )
        X_Beta = np.exp(X @ init['Beta']).T
        X_Beta = pd.DataFrame(
            restrict_X_Beta(X_Beta),
            index=InitProp['internal_vars']['gene_list_reg'],
            columns=X.index
        )
    else:
        cellWeight = 1
        X_Beta = pd.DataFrame(
            np.ones((len(InitProp['internal_vars']['gene_list_reg']), cell_locations.shape[0])),
            index=InitProp['internal_vars']['gene_list_reg'],
            columns=cell_locations.index
        )

    alpha = weights.sum(1)
    spot_label = sp_index_table.loc[cell_locations[spot_index_name].values].values.squeeze()
    nUMI = InitProp['reconSpatialRNA']['nUMI'].loc[sp_index].values.squeeze()
    signature_matrix = InitProp['cell_type_info']['renorm']['cell_type_means'].loc[
        InitProp['internal_vars']['gene_list_reg'],
        cell_type_names
    ]

    if P is not None:
        cell_num_total_P = np.asarray(P.sum(axis=1)).squeeze()
        cell_num_total = np.array([
            np.where(cell_locations[spot_index_name].values == sp_name)[0].shape[0]
            for sp_name in sp_index
        ])
    else:
        cell_num_total = np.ones((cell_locations.shape[0],))
        cell_num_total_P = cell_num_total
        P = diags([1] * cell_locations.shape[0], 0, format="csr")

    weights_long = weights[spot_label, :]
    pos_celltype = []
    for i in range(weights.shape[0]):
        thresh = min(weights[i].sum() / (2 * cell_num_total.max()), weights[i].sum() / 5)
        candidates = np.where(weights[i] > thresh)[0]
        if candidates.shape[0] == 0:
            candidates = np.array([0, 1, 2])
        pos_celltype.append(candidates)

    com = []
    for i in np.arange(2, cell_num_total.max() + 1):
        com.append(list(itertools.combinations(range(i), 2)))

    init['k'] = np.argmax(weights_long, axis=-1)

    sigma = round(InitProp['internal_vars']['sigma'] * 100)
    puck = InitProp['reconSpatialRNA']
    MIN_UMI = InitProp['config']['UMI_min_sigma']

    puck_counts = puck['counts'].loc[:, sp_index]
    puck_nUMI = puck['nUMI'].loc[sp_index]
    N_fit = min(InitProp['config']['N_fit'], (puck_nUMI > MIN_UMI).sum().item())

    if N_fit == 0:
        raise ValueError(
            'choose_sigma_c determined a N_fit of 0! Try decreasing UMI_min_sigma (currently {}).'
            .format(MIN_UMI)
        )

    fit_ind = np.random.choice(puck_nUMI[puck_nUMI > MIN_UMI].index, N_fit, replace=False)
    beads = puck_counts.loc[InitProp['internal_vars']['gene_list_reg'], fit_ind].values.T
    loggings.info('chooseSigma: using initial Q_mat with sigma = {}'.format(sigma / 100))
    puck_nUMI = puck_nUMI * alpha[:, None]

    for epoch in range(n_epoch):
        with FileLock(process_log + '.lock'):
            with open(process_log, 'w') as f:
                f.write(str(30 + int((epoch + 1) / n_epoch * 50)))

        likelihood_vars = {
            'Q_mat': Q_mat_all[str(sigma)],
            'X_vals': X_vals_loc,
            'N_X': Q_mat_all[str(sigma)].shape[1],
            'K_val': Q_mat_all[str(sigma)].shape[0] - 3
        }

        print("Updating cell labels!")
        inp_args = []
        for i in np.random.choice(np.arange(sp_index.shape[0]), sp_index.shape[0], replace=False):
            index = np.where(spot_label == i)[0]
            Y = puck_counts.loc[InitProp['internal_vars']['gene_list_reg'], sp_index[i]].values
            inp_args.append((Y, index, pos_celltype[i], nUMI[i], alpha[i], cell_num_total_P[i], nu, i))

        xbeta_ref = ray.put(X_Beta.values)
        init_update_res = ray.get([UpdateCellLabel_Greedy.remote(args, xbeta_ref) for args in inp_args])

        for init_update_index, init_update in init_update_res:
            init['k'][init_update_index] = init_update

        if MH:
            print("MCMC starts!")
            res = run_MH_full(
                puck_counts.loc[InitProp['internal_vars']['gene_list_reg'], :].values,
                X.values,
                puck_nUMI.values,
                init['Beta'].values,
                init['Gamma'].values,
                likelihood_vars,
                P,
                signature_matrix.loc[:, [cell_type_names[ct] for ct in init['k']]].values,
                device
            )
            init['Beta'] = pd.DataFrame(res['beta'], index=X.columns, columns=InitProp['internal_vars']['gene_list_reg'])
            init['Gamma'] = pd.DataFrame(res['gamma'], index=X.columns, columns=InitProp['internal_vars']['gene_list_reg'])
            X_Beta = np.exp(X @ init['Beta']).T
            X_Beta = pd.DataFrame(
                restrict_X_Beta(X_Beta),
                index=InitProp['internal_vars']['gene_list_reg'],
                columns=X.index
            )

        lambda_hat = calc_lambda_hat(
            P,
            signature_matrix.loc[:, [cell_type_names[ct] for ct in init['k']]].values,
            X_Beta.values,
            MH
        )
        lambda_hat = pd.DataFrame(lambda_hat, index=InitProp['internal_vars']['gene_list_reg'], columns=sp_index)
        prediction = lambda_hat.loc[:, fit_ind] * puck_nUMI.loc[fit_ind].values.squeeze()[None, :]
        print('Likelihood value: {}'.format(
            calc_log_l_vec(prediction.values.T.reshape(-1), beads.reshape(-1), likelihood_vars=likelihood_vars)
        ))

        sigma_prev = sigma
        sigma = chooseSigma(prediction, beads.T, Q_mat_all, likelihood_vars['X_vals'], sigma)
        loggings.info('Sigma value: {}'.format(sigma / 100))

        if sigma_prev == sigma and epoch > 1:
            break

    InitProp['internal_vars']['sigma'] = sigma / 100
    if MH:
        InitProp['internal_vars']['Gamma'] = init['Gamma']
        InitProp['internal_vars']['Beta'] = init['Beta']
    InitProp['internal_vars']['Q_mat'] = Q_mat_all[str(sigma)]
    InitProp['internal_vars']['X_vals'] = likelihood_vars['X_vals']
    cell_locations['discrete_label'] = init['k']
    InitProp['discrete_label'] = cell_locations

    if hs_ST:
        try:
            InitProp['label2ct'] = pd.DataFrame(
                InitProp['results']['weights'].columns,
                index=np.arange(InitProp['results']['weights'].shape[1])
            )
        except Exception:
            InitProp['label2ct'] = pd.DataFrame(
                InitProp['results'].columns,
                index=np.arange(InitProp['results'].shape[1])
            )
    else:
        InitProp['label2ct'] = pd.DataFrame(
            InitProp['results'].columns,
            index=np.arange(InitProp['results'].shape[1])
        )

    return InitProp


def SmoothPrior(i, j, init_k, j_vector, index_of_j, nu, cellWeight, logp=False):
    """
    Smoothness prior encouraging neighboring cells to share labels.
    """
    init_fake = init_k.copy()
    init_fake[index_of_j] = j
    if isinstance(cellWeight, (int, float)):
        U = -((init_fake[j_vector] != i) * cellWeight).sum() * nu / j_vector.shape[0]
    else:
        U = -((init_fake[j_vector] != i) * cellWeight[index_of_j, j_vector].toarray()).sum() * nu / j_vector.shape[0]
    return U if logp else np.exp(U) 