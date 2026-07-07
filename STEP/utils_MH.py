import numpy as np
from numpy.random import normal, rand
from scipy.special import gammaln
from torch.distributions.normal import Normal
import pandas as pd
from collections import Counter
from torch_sparse import SparseTensor
from scipy.sparse import issparse
from tqdm import tqdm
from .utils import *
from .utils_pyRCTD import *


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch_sparse.SparseTensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return SparseTensor(
        row=indices[0], col=indices[1], value=values, sparse_sizes=shape
    )


def get_spatial_matrix(
    T_spot,
    T_cell,
    spot_diameter,
    radius,
    polygon,
    spot_index_name,
    max_cell_number=10,
):
    """
    Build spatial overlap matrix P (spot x cell) and a cell_locations table.

    P[i, j] = overlap area between cell j and spot i
    (computed via `intersection_area`).
    """
    n_cell = T_cell.shape[0]
    n_spot = T_spot.shape[0]
    index_cell = T_cell.index.values
    index_spot = T_spot.index.values

    # P: rows = spots, columns = cells
    P = pd.DataFrame(
        np.zeros((n_spot, n_cell)), index=index_spot, columns=index_cell
    )

    # cell_locations: which spot each cell is finally assigned to
    cell_locations = pd.DataFrame(columns=[spot_index_name])

    # First pass: for each cell, find nearest spot within radius and assign area
    for i in range(n_cell):
        dist_matrix = vectorized_pdist(T_cell.iloc[i, :], T_spot)
        min_dist_cur = np.min(dist_matrix)
        if min_dist_cur <= spot_diameter / 2 + radius[i]:
            temp = np.where(dist_matrix == min_dist_cur)[0]
            P.iloc[temp, i] = intersection_area(
                T_spot.iloc[temp, :], spot_diameter / 2, polygon[i]
            )

    # Second pass: enforce max_cell_number per spot and fill cell_locations
    for index, row in P.iterrows():
        cells = np.where(row > 0)[0]
        if len(cells) >= max_cell_number:
            cells_final = np.random.choice(
                cells, max_cell_number, replace=False
            )
            # drop extra cells on this spot
            P.loc[index, index_cell[cells[~np.isin(cells, cells_final)]]] = 0
            # record kept cells
            for cell in cells_final:
                cell_locations.loc[index_cell[cell]] = index
        else:
            for cell in cells:
                cell_locations.loc[index_cell[cell]] = index

    # Add coordinates to cell_locations (2D or 3D)
    if T_cell.shape[1] == 2:
        cell_locations[["x", "y"]] = T_cell.loc[cell_locations.index]
    else:
        cell_locations[["x", "y", "z"]] = T_cell.loc[cell_locations.index]

    return P, cell_locations


def get_partition_from_cell_locations(
    cell_locations,
    T_spot,
    spot_diameter,
    polygon,
    spot_index_name,
    max_cell_number=10,
):
    """
    Build partition matrix P from pre-existing cell_locations.

    If a spot has more than max_cell_number cells, randomly sample
    max_cell_number cells and discard the rest.
    P values are computed via intersection_area (same as get_spatial_matrix).

    Returns (P, cell_locations) where P is a spot x cell DataFrame
    and cell_locations is filtered.
    """
    index_spot = T_spot.index.values

    kept_cells = []
    for spot in index_spot:
        cells_in_spot = cell_locations[cell_locations[spot_index_name] == spot].index.values
        if len(cells_in_spot) > max_cell_number:
            cells_in_spot = np.random.choice(cells_in_spot, max_cell_number, replace=False)
        kept_cells.extend(cells_in_spot)

    cell_locations = cell_locations.loc[kept_cells]

    n_spot = T_spot.shape[0]
    n_cell = cell_locations.shape[0]
    P = pd.DataFrame(
        np.zeros((n_spot, n_cell)), index=index_spot, columns=cell_locations.index
    )

    for cell_id in cell_locations.index:
        spot_id = cell_locations.loc[cell_id, spot_index_name]
        if spot_id in P.index:
            P.loc[spot_id, cell_id] = intersection_area(
                T_spot.loc[spot_id].values, spot_diameter / 2, polygon[cell_id]
            )

    return P, cell_locations


def get_spatial_matrix_segmentation(cell_locations, sp_index_table, spot_index_name):
    """
    Build a binary spot x cell assignment matrix G from pre-defined
    segmentation labels in `cell_locations[spot_index_name]`.
    """
    n_cell = cell_locations.shape[0]
    n_spot = sp_index_table.shape[0]

    G = pd.DataFrame(
        np.zeros((n_spot, n_cell)),
        index=sp_index_table.index,
        columns=cell_locations.index,
    )
    for i in cell_locations.index:
        G.loc[cell_locations.loc[i, spot_index_name], i] = 1
    return G.to_numpy()


def get_dummy(Z, class_name=None):
    """
    One-hot encode a label vector Z into a dummy matrix (DataFrame).
    """
    if class_name is None:
        class_name = np.unique(Z)

    Z_dummy = np.zeros((len(Z), len(class_name)))
    Z_dummy = pd.DataFrame(Z_dummy, columns=class_name)

    for i, class_name_i in enumerate(class_name):
        Z_dummy[class_name_i] = (Z == class_name_i).astype(int)

    return Z_dummy


def run_MH_single(
    counts,
    X,
    nUMI,
    beta_initial,
    gamma_initial,
    likelihood_vars,
    P,
    cell_signature_matrix,
):
    """
    Wrapper for RunMH_single: run MH, then return posterior mean beta, gamma
    with hard thresholding on gamma.
    """
    sigma_beta = 1
    res = RunMH_single(
        counts,
        X,
        nUMI,
        P,
        sigma_beta,
        beta_initial,
        gamma_initial,
        likelihood_vars,
        cell_signature_matrix,
    )

    iter_ = res["beta"].shape[0]

    beta = np.mean(res["beta"][iter_ // 2 :, :], axis=0)
    gamma = np.mean(res["gamma"][iter_ // 2 :, :], axis=0)

    # threshold gamma and zero-out beta where gamma is small
    select_index = gamma < 0.1
    beta[select_index] = 0
    gamma[select_index] = 0
    gamma[~select_index] = 1

    return {"beta": beta, "gamma": gamma}


def run_MH_full(
    counts,
    X,
    nUMI,
    beta_initial,
    gamma_initial,
    likelihood_vars,
    P,
    cell_signature_matrix,
    device,
):
    """
    Wrapper for RunMH_full (vectorized / GPU): return posterior mean beta, gamma
    with hard thresholding on gamma.
    """
    sigma_beta = 1
    res = RunMH_full(
        counts,
        X,
        nUMI,
        P,
        sigma_beta,
        beta_initial,
        gamma_initial,
        likelihood_vars,
        cell_signature_matrix,
        device,
    )

    beta = res["beta"]
    gamma = res["gamma"]

    # stricter threshold than single version
    select_index = gamma < 0.05
    beta[select_index] = 0
    gamma[select_index] = 0
    gamma[~select_index] = 1

    return {"beta": beta, "gamma": gamma}


def dnorm_functions_single(r, mu, sigma):
    """
    Univariate normal pdf evaluated at r for N(mu, sigma^2).
    """
    return (
        np.exp(-0.5 * ((r - mu) / sigma) ** 2)
        / (sigma * np.sqrt(2 * np.pi))
    )


def RunMH_single(
    y,
    X,
    s,
    P,
    sigma_beta,
    beta_initial,
    gamma_initial,
    likelihood_vars,
    cell_signature_matrix,
):
    """
    Scalar MH sampler for a single gene / single chain.
    Update beta and gamma coordinate-wise.
    """
    M, Q = X.shape
    N = P.shape[0]

    tau_beta = 1

    # MCMC settings
    iter = 1000
    burn = iter // 2  # burn used outside via slicing

    beta_store = np.zeros((iter, Q))
    gamma_store = np.zeros((iter, Q))

    a_gamma = 0.5
    b_gamma = 0.5

    beta = beta_initial.reshape((Q, 1))
    gamma = np.array(gamma_initial)
    cell_signature_matrix = cell_signature_matrix.reshape((1, M))

    # initial intensity
    lambda_ = calc_lambda_hat(
        P, cell_signature_matrix, np.exp(X @ beta).T
    ).T * s

    for i in range(iter):
        # update each beta_q, gamma_q
        for q in range(Q):
            beta_temp = beta.copy()

            # propose flipping gamma_q
            if gamma[q] == 0:
                gamma_temp = 1
                beta_temp[q, 0] = normal(0, tau_beta, 1)[0]
            else:
                gamma_temp = 0
                beta_temp[q, 0] = 0.0

            x_beta = restrict_X_Beta(np.exp(X @ beta_temp).T)
            lambda_temp = calc_lambda_hat(
                P, cell_signature_matrix, x_beta
            ).T * s

            # log MH ratio for gamma flip
            hastings = calc_log_l_vec(
                lambda_.reshape(-1),
                y.reshape(-1),
                likelihood_vars=likelihood_vars,
            ) - calc_log_l_vec(
                lambda_temp.reshape(-1),
                y.reshape(-1),
                likelihood_vars=likelihood_vars,
            )

            if gamma[q] == 0:
                # 0 -> 1
                hastings += np.log(
                    dnorm_functions_single(
                        beta_temp[q, 0], 0.0, sigma_beta
                    )
                )
                hastings += (
                    gammaln(a_gamma + gamma_temp)
                    + gammaln(b_gamma + 2 - gamma_temp)
                    - gammaln(1 + gamma_temp)
                    - gammaln(2 - gamma_temp)
                )
                hastings -= (
                    gammaln(a_gamma + gamma[q])
                    + gammaln(b_gamma + 2 - gamma[q])
                    - gammaln(1 + gamma[q])
                    - gammaln(2 - gamma[q])
                )
                hastings -= np.log(
                    dnorm_functions_single(
                        beta_temp[q, 0], 0.0, tau_beta
                    )
                )
            else:
                # 1 -> 0
                hastings -= np.log(
                    dnorm_functions_single(
                        beta[q, 0], 0.0, sigma_beta
                    )
                )
                hastings += (
                    gammaln(a_gamma + gamma_temp)
                    + gammaln(b_gamma + 2 - gamma_temp)
                    - gammaln(1 + gamma_temp)
                    - gammaln(2 - gamma_temp)
                )
                hastings -= (
                    gammaln(a_gamma + gamma[q])
                    + gammaln(b_gamma + 2 - gamma[q])
                    - gammaln(1 + gamma[q])
                    - gammaln(2 - gamma[q])
                )
                hastings += np.log(
                    dnorm_functions_single(
                        beta[q, 0], 0.0, tau_beta
                    )
                )

            # accept / reject gamma flip
            if hastings > np.log(rand()):
                gamma[q] = gamma_temp

            # conditional beta update
            if gamma[q] == 0:
                beta[q, 0] = 0.0
                x_beta = restrict_X_Beta(np.exp(X @ beta).T)
                lambda_ = calc_lambda_hat(
                    P, cell_signature_matrix, x_beta
                ).T * s
            else:
                beta_temp[q, 0] = normal(
                    beta[q, 0], tau_beta / 2, 1
                )[0]
                x_beta = restrict_X_Beta(np.exp(X @ beta_temp).T)
                lambda_temp = calc_lambda_hat(
                    P, cell_signature_matrix, x_beta
                ).T * s

                hastings = calc_log_l_vec(
                    lambda_.reshape(-1),
                    y.reshape(-1),
                    likelihood_vars=likelihood_vars,
                ) - calc_log_l_vec(
                    lambda_temp.reshape(-1),
                    y.reshape(-1),
                    likelihood_vars=likelihood_vars,
                )
                hastings = hastings + (
                    -beta_temp[q, 0]
                    * beta_temp[q, 0]
                    / 2
                    / sigma_beta
                    / sigma_beta
                )
                hastings = hastings - (
                    -beta[q, 0]
                    * beta[q, 0]
                    / 2
                    / sigma_beta
                    / sigma_beta
                )

                if hastings > np.log(rand()):
                    beta[q, 0] = beta_temp[q, 0].copy()
                    lambda_ = lambda_temp.copy()

        beta_store[i, :] = beta[:, 0]
        gamma_store[i, :] = gamma

    return {"beta": beta_store, "gamma": gamma_store}


def RunMH_full(
    Y,
    X,
    s_,
    P,
    sigma_beta,
    beta_initial,
    gamma_initial,
    likelihood_vars_,
    cell_signature_matrix_,
    device,
):
    """
    Vectorized / GPU MH sampler for all factors x genes.
    """
    # move inputs to torch / device
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    beta = torch.tensor(beta_initial, dtype=torch.float32).to(device)
    gamma = torch.tensor(gamma_initial, dtype=torch.float32).to(device)
    cell_signature_matrix = torch.tensor(
        cell_signature_matrix_.T, dtype=torch.float32
    ).to(device)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    if issparse(P):
        P = sparse_mx_to_torch_sparse_tensor(P).to(device)
    else:
        P = torch.tensor(P, dtype=torch.float32).to(device)

    s = torch.tensor(s_, dtype=torch.float32).to(device).reshape(-1)

    likelihood_vars = likelihood_vars_.copy()
    likelihood_vars["Q_mat"] = torch.tensor(
        likelihood_vars["Q_mat"], dtype=torch.float32
    ).to(device)
    likelihood_vars["X_vals"] = torch.tensor(
        likelihood_vars["X_vals"], dtype=torch.float32
    ).to(device)

    y = Y.T  # (spot x gene)
    spot, gene = y.shape
    factor = X.shape[1]
    tau_beta = 1

    # MCMC settings
    iter = 1500
    burn = iter // 2

    a_gamma = 0.5
    b_gamma = 0.5

    # precompute rescaled P for lambda = fast_M @ (sig * x_beta)
    fast_M = P / (P.sum(dim=1) / s)[:, None]

    counter = 0
    sum_beta = torch.zeros_like(beta)
    sum_gamma = torch.zeros_like(gamma)

    distribution_sigma = Normal(0.0, sigma_beta)
    distribution_tau = Normal(0.0, tau_beta)

    for i in tqdm(range(iter)):
        # current x_beta and lambda
        x_beta = restrict_X_Beta(torch.exp(X @ beta), GPU=True)
        lambda_ = fast_M @ (cell_signature_matrix * x_beta)

        # ----- block-update gamma and beta jointly -----
        beta_temp = beta.clone()
        gamma_temp = 1 - gamma  # flip {0,1}

        select_index = gamma_temp != 0
        beta_temp[~select_index] = 0.0
        beta_temp[select_index] = torch.normal(
            0, tau_beta, size=(select_index.sum(),), device=device
        )

        x_beta = restrict_X_Beta(torch.exp(X @ beta_temp), GPU=True)
        lambda_temp = fast_M @ (cell_signature_matrix * x_beta)

        # log-likelihood difference (vector over factors x genes)
        hastings = calc_log_l_vec(
            lambda_, y, likelihood_vars=likelihood_vars, return_vec=True, GPU=True
        ).sum(dim=0)
        hastings -= calc_log_l_vec(
            lambda_temp, y, likelihood_vars=likelihood_vars, return_vec=True, GPU=True
        ).sum(dim=0)

        # prior for gamma (Beta-Bernoulli form)
        hastings = hastings + (
            torch.lgamma(a_gamma + gamma_temp)
            + torch.lgamma(b_gamma + 2 - gamma_temp)
            - torch.lgamma(1 + gamma_temp)
            - torch.lgamma(2 - gamma_temp)
        )
        hastings = hastings - (
            torch.lgamma(a_gamma + gamma)
            + torch.lgamma(b_gamma + 2 - gamma)
            - torch.lgamma(1 + gamma)
            - torch.lgamma(2 - gamma)
        )

        # normal priors and proposal terms for beta (on flipped entries)
        hastings = hastings + distribution_sigma.log_prob(beta_temp) * select_index
        hastings = hastings - distribution_tau.log_prob(beta_temp) * select_index
        hastings = hastings - distribution_sigma.log_prob(beta) * (~select_index)
        hastings = hastings + distribution_tau.log_prob(beta) * (~select_index)

        # accept/reject gamma flip per (factor, gene)
        update_index = hastings > torch.log(
            torch.rand((factor, gene), device=device)
        )
        gamma[update_index] = gamma_temp[update_index]

        # ----- conditional beta update given new gamma -----
        select_index = gamma == 0
        beta[select_index] = 0.0

        x_beta = restrict_X_Beta(torch.exp(X @ beta), GPU=True)
        lambda_ = fast_M @ (cell_signature_matrix * x_beta)

        beta_temp[~select_index] = torch.normal(
            beta[~select_index], tau_beta / 2
        )
        beta_temp[select_index] = 0.0

        x_beta = restrict_X_Beta(torch.exp(X @ beta_temp), GPU=True)
        lambda_temp = fast_M @ (cell_signature_matrix * x_beta)

        hastings = calc_log_l_vec(
            lambda_, y, likelihood_vars=likelihood_vars, return_vec=True, GPU=True
        ).sum(dim=0)
        hastings -= calc_log_l_vec(
            lambda_temp, y, likelihood_vars=likelihood_vars, return_vec=True, GPU=True
        ).sum(dim=0)

        hastings = hastings + (
            -beta_temp * beta_temp / 2 / sigma_beta / sigma_beta
        )
        hastings = hastings - (
            -beta * beta / 2 / sigma_beta / sigma_beta
        )

        update_index = hastings > torch.log(
            torch.rand((factor, gene), device=device)
        )
        beta[update_index] = beta_temp[update_index]

        # accumulate posterior means after burn-in
        if i >= burn:
            counter += 1
            sum_beta += beta
            sum_gamma += gamma

    return {
        "beta": (sum_beta / counter).detach().cpu().numpy(),
        "gamma": (sum_gamma / counter).detach().cpu().numpy(),
    }


def vectorized_pdist(A, B):
    """
    Compute Euclidean distance from 1D point A to each row of B (vectorized).
    """
    an = np.sum(A ** 2)
    bn = np.sum(B ** 2, axis=1)
    AB = np.dot(A, B.T)
    dist = np.sqrt(an + bn - 2 * AB)
    return dist


def calc_lambda_hat(Partion, cell_signature_matrix, Impact_factors, MH=True):
    """
    Compute lambda_hat = (cell_signature_matrix * Impact_factors) @ Partion.T
    optionally normalized by row sums of Partion.
    """
    if MH:
        lambda_hat = (cell_signature_matrix * Impact_factors) @ Partion.T
        if issparse(Partion):
            lambda_hat = lambda_hat / np.array(Partion.sum(axis=1))[None, :].squeeze(
                -1
            )
        else:
            lambda_hat = lambda_hat / Partion.sum(axis=1)[None, :]
    else:
        lambda_hat = cell_signature_matrix * Impact_factors
    return lambda_hat


def calc_log_l3(Gamma, pie_gamma=None):
    """
    Log prior for Gamma:
      - If pie_gamma is None: Beta-Bernoulli style prior with (a,b) = (0.5,0.5)
      - Else: fixed mixture prior using Bernoulli(pie_gamma).
    """
    if pie_gamma is None:
        a_gamma = 0.5
        b_gamma = 0.5
        return (
            gammaln(a_gamma + Gamma)
            + gammaln(b_gamma + 1 - Gamma)
            + gammaln(a_gamma + b_gamma)
            - gammaln(a_gamma + b_gamma + 1)
            - gammaln(a_gamma)
            - gammaln(b_gamma)
        ).values.sum()
    else:
        Gamma_0 = (Gamma == 0).astype(int) * (1 - pie_gamma)
        Gamma_1 = (Gamma == 1).astype(int) * pie_gamma
        return -np.log(Gamma_0 + Gamma_1).values.sum()


def calc_log_l2(Gamma, Beta, sigma_beta=1):
    """
    Log prior contribution for Beta given Gamma under N(0, sigma_beta^2)
    and variable selection (Beta * Gamma).
    """
    return -np.log(dnorm_functions_single(Beta, 0, sigma_beta) * Gamma).values.sum()


def restrict_X_Beta(x_beta, f_thresh=0, f_max=5, GPU=False):
    """
    Clamp the elements of x_beta into [f_thresh, f_max].
    """
    if GPU:
        x_beta = torch.clamp(x_beta, min=f_thresh, max=f_max)
    else:
        x_beta = np.clip(x_beta, f_thresh, f_max)

    return x_beta