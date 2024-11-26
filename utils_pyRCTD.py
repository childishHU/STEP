import sys
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from numpy import array, dot
from qpsolvers import solve_qp
import qpsolvers
import itertools
from CVAE import * 
# from qpsolvers.solvers.qpswift_ import qpswift_solve_qp
import psutil
import ray
import torch
data_type = 'float32'
import os
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')




def run_RCTD(RCTD, Q_mat_all, X_vals_loc, out_dir,doublet_mode = 'full', hs_ST=False, loggings = None, drop=False):
    if drop:
        matrix = RCTD['spatialRNA']['counts'].loc[RCTD['internal_vars']['gene_list_reg']].values
        flattened = matrix.flatten()
        sorted_indices = np.argsort(flattened)
        sparse = 0.45
        total_elements = matrix.shape[0] * matrix.shape[1]
        keep_top_n = int((1 - sparse) * total_elements)
        flattened_sparse = flattened.copy()
        flattened_sparse[sorted_indices[:-keep_top_n]] = 0
        sparse_matrix = flattened_sparse.reshape(matrix.shape)
        RCTD['spatialRNA']['counts'].loc[RCTD['internal_vars']['gene_list_reg']] = sparse_matrix
        # RCTD['internal_vars']['gene_list_reg'] = np.random.choice(RCTD['internal_vars']['gene_list_reg'], int(len(RCTD['internal_vars']['gene_list_reg']) * 0.9), replace=False)
    #RCTD['reconSpatialRNA'] = RCTD['spatialRNA'].copy()
    RCTD = effectBalance(RCTD, out_dir, hs_ST,loggings = loggings)
    RCTD = fitBulk(RCTD, loggings = loggings)
    RCTD = choose_sigma_c(RCTD, Q_mat_all, X_vals_loc, loggings = loggings)
    RCTD = fitPixels(RCTD, doublet_mode = doublet_mode, loggings = loggings)
    return RCTD

def fitBulk(RCTD, loggings = None):
    bulkData = prepareBulkData(RCTD['cell_type_info']['info']['cell_type_means'], RCTD['reconSpatialRNA'], RCTD['internal_vars']['gene_list_bulk'])
    loggings.info('fitBulk: decomposing bulk')
    decompose_results = decompose_full(bulkData['X'],RCTD['reconSpatialRNA']['nUMI'].sum().item(),
                                      np.array(bulkData['b']), verbose = False, constrain = False, MIN_CHANGE = RCTD['config']['MIN_CHANGE_BULK'],
                                      n_iter = 100, bulk_mode = True, loggings = loggings)
    RCTD['internal_vars']['proportions'] = decompose_results['weights']
    RCTD['cell_type_info']['renorm'] = RCTD['cell_type_info']['info'].copy()
    RCTD['cell_type_info']['renorm']['cell_type_means'] = get_norm_ref(RCTD['reconSpatialRNA'], RCTD['cell_type_info']['info']['cell_type_means'], RCTD['internal_vars']['gene_list_bulk'], decompose_results['weights'])
    return RCTD

def get_norm_ref(puck, cell_type_means, gene_list, proportions):
    bulk_vec = puck['counts'].sum(1)
    weight_avg = (cell_type_means.loc[gene_list, :] * (proportions / proportions.sum()).values.squeeze()).sum(1)
    weight_avg = np.where(weight_avg == 0, 1e-6, weight_avg)
    target_means = bulk_vec.loc[gene_list] / puck['nUMI'].sum().item()
    cell_type_means_renorm = cell_type_means.loc[gene_list,:] / (weight_avg / target_means).values.squeeze()[:, None]
    return cell_type_means_renorm


def prepareBulkData(cell_type_means, puck, gene_list, MIN_OBS = 10):
    bulk_vec = puck['counts'].sum(1)
    gene_list = np.array(list(set(bulk_vec.index[bulk_vec >= MIN_OBS]) & set(gene_list)))
    nUMI = puck['nUMI'].sum().item()
    X = cell_type_means.loc[gene_list,:] * nUMI
    b = bulk_vec[gene_list]
    ret = {'X': X, 'b': b}
    return ret

def effectBalance(RCTD, out_dir ,hs_ST=False, loggings=None):

    def mse_loss(array1, array2):
        return np.mean((array1 - array2) ** 2)

    args = Args()
    

    if hs_ST:
        scaler = 1e3
        args.k_st_max = args.k_sc_max
        args.num_hidden_layer = 2
        n_celltype = 150
    else:
        scaler = 1e4
        n_celltype = RCTD['cell_type_info']['info']['n_cell_types']
        
        
    ref = RCTD['reference']['counts'].loc[RCTD['internal_vars']['gene_list_bulk']].T.copy()
    ref = ref / ref.sum(axis=1).values[:, None]
    mat_sc = ref.values * scaler

    spa = RCTD['spatialRNA']['counts'].T.copy()
    spa = spa / spa.sum(axis=1).values[:, None]
    mat_st = spa.values * scaler 

    if os.path.exists(os.path.join(out_dir, 'model/model.pth')):

        loggings.info('Loading existing model, no need to train')
        cvae = torch.load(os.path.join(out_dir, 'model/model.pth'))

        p = len(RCTD['internal_vars']['gene_list_bulk'])
        p_cond = 1
        latent_dim = n_celltype * 3
        hidden_dim = list(np.floor(np.geomspace(latent_dim, p, args.num_hidden_layer+2)[1:args.num_hidden_layer+1]).astype('int'))

        encoder = Encoder(p, p_cond, latent_dim, hidden_dim[::-1], use_batch_norm=args.use_batch_norm).to(args.device)
        decoder = Decoder(p, p_cond, latent_dim, hidden_dim, use_batch_norm=args.use_batch_norm).to(args.device)
        encoder.load_state_dict(cvae.encoder.state_dict())
        decoder.load_state_dict(cvae.decoder.state_dict())
        encoder.eval()
        decoder.eval()

        minmax_sc = MinMaxScaler(feature_range=(0, args.input_max))
        mat_sc_r = np.log1p(mat_sc)
        mat_sc_r = minmax_sc.fit_transform(mat_sc_r)

        minmax_st = MinMaxScaler(feature_range=(0, args.input_max))
    
        mat_sp_r = np.log1p(mat_st)
        mat_sp_r = minmax_st.fit_transform(mat_sp_r)

        new_st, new_sc = Reconstruct(encoder, decoder, mat_sc_r, mat_sp_r, minmax_sc, args)

    else:
        loggings.info('Training CVAE!')
        model, new_st, new_sc = DomainAdaptation(mat_st, mat_sc, n_celltype, out_dir, args, loggings)
        torch.save(model, os.path.join(out_dir, 'model/model.pth'))
    

    loggings.info('sc_loss: {}'.format(mse_loss(new_sc ,mat_sc)))
    loggings.info('st_loss: {}'.format(mse_loss(new_st ,mat_st)))
    

    new_st = np.rint((new_st / new_st.sum(axis=1)[:, None]).T * RCTD['spatialRNA']['counts'].sum(axis=0).values[None, :])
    #new_st = np.rint(new_st.T)
    new_st = pd.DataFrame(new_st, columns=RCTD['spatialRNA']['counts'].columns, index=RCTD['internal_vars']['gene_list_bulk'])


    RCTD['reconSpatialRNA'] = RCTD['spatialRNA'].copy()
    RCTD['reconSpatialRNA']['counts'] = new_st
    #RCTD['reconSpatialRNA']['nUMI'] = pd.DataFrame([1e4] * RCTD['reconSpatialRNA']['nUMI'].shape[0], index=RCTD['reconSpatialRNA']['nUMI'].index)
    
    return RCTD

def calc_Q_all(x, bead, likelihood_vars = None):
    X_vals = likelihood_vars['X_vals']
    Q_mat = likelihood_vars['Q_mat']
    epsilon = 1e-4
    X_max = max(X_vals)
    delta = 1e-5
    x = np.minimum(np.maximum(epsilon, x),X_max - epsilon)
    l = np.floor((x / delta) ** (2 / 3))
    l = np.minimum(l, 900) + np.floor(np.maximum(l - 900,0) / 30)
    l = l - 1 #change index to python
    l = l.astype(int)
    prop = (X_vals[l + 1] - x) / (X_vals[l+1] - X_vals[l])
    v1 = Q_mat[np.floor(bead).astype(int),l + 1]
    k = Q_mat[np.floor(bead).astype(int), l] - v1
    r1 = k * prop + v1
    v1 = Q_mat[np.floor(bead).astype(int) + 1, l+1]
    k = Q_mat[np.floor(bead).astype(int) + 1, l] - v1
    r2 = k * prop + v1
    v1 = Q_mat[np.floor(bead).astype(int) + 2, l+1]
    k = Q_mat[np.floor(bead).astype(int) + 2, l] - v1
    r3 = k * prop + v1
    return {'r1': r1,'r2': r2,'r3': r3}

def get_d1_d2(B, prediction, likelihood_vars = None):
    X_vals = likelihood_vars['X_vals']
    Q_mat = likelihood_vars['Q_mat']
    K_val = likelihood_vars['K_val']
    bead = B
    epsilon = 1e-4
    X_max = max(X_vals)
    x = np.minimum(np.maximum(epsilon, prediction), X_max - epsilon)
    Q_cur = calc_Q_all(x, bead, likelihood_vars = likelihood_vars)
    bead[bead > K_val] = K_val
    Q_k = Q_cur['r1']
    Q_k1 = Q_cur['r2']
    Q_k2 = Q_cur['r3']
    Q_d1 = 1/x * (-(bead+1)*Q_k1 + bead*Q_k)
    Q_d2 = 1/(x**2)*((bead+1)*(bead+2)*Q_k2 - bead*(2*(bead+1)*Q_k1 - (bead-1)*Q_k))
    d1_vec = Q_d1 / Q_k
    d2_vec = -Q_d1**2/(Q_k**2) + Q_d2/Q_k
    return {'d1_vec': d1_vec, 'd2_vec': d2_vec}


def calc_Q_k(x, bead, likelihood_vars = None):
    X_vals = likelihood_vars['X_vals']
    Q_mat = likelihood_vars['Q_mat']
    K_val = likelihood_vars['K_val']
    bead = np.copy(bead)
    bead[bead > K_val] = K_val
    epsilon = 1e-4
    X_max = max(X_vals)
    delta = 1e-5
    x = np.minimum(np.maximum(epsilon, x), X_max - epsilon)
    l = np.floor((x / delta) ** (2 / 3))
    l = np.minimum(l, 900) + np.floor(np.maximum(l - 900,0) / 30)
    l = l - 1 #change index to python
    l = l.astype(int)
    prop = (X_vals[l + 1] - x) / (X_vals[l+1] - X_vals[l])
    v1 = Q_mat[np.floor(bead).astype(int),l + 1]
    k = Q_mat[np.floor(bead).astype(int), l] - v1
    r1 = k * prop + v1
    return r1

def calc_Q_k_GPU(x, bead, likelihood_vars=None):
    X_vals = likelihood_vars['X_vals']
    Q_mat = likelihood_vars['Q_mat']
    K_val = likelihood_vars['K_val']
    bead = bead.clone()
    bead[bead > K_val] = K_val
    epsilon = 1e-4
    X_max = X_vals.max()
    delta = 1e-5
    x = torch.clamp(x, epsilon, X_max - epsilon)
    l = torch.floor((x / delta) ** (2 / 3))
    l = torch.minimum(l, torch.tensor(900)) + torch.floor(torch.maximum(l - 900, torch.tensor(0)) / 30)
    l = l - 1  # change index to python
    l = l.int()
    prop = (X_vals[l + 1] - x) / (X_vals[l + 1] - X_vals[l])
    v1 = Q_mat[torch.floor(bead).long(), l + 1]
    k = Q_mat[torch.floor(bead).long(), l] - v1
    r1 = k * prop + v1
    return r1

def calc_log_l_vec(lamb, Y, return_vec = False, likelihood_vars = None, GPU = False):
    if GPU:
        log_l_vec = -torch.log(calc_Q_k_GPU(lamb,Y, likelihood_vars = likelihood_vars))
    else:    
        log_l_vec = -np.log(calc_Q_k(lamb,Y, likelihood_vars = likelihood_vars))
    if return_vec:
        return log_l_vec
    else:
        return log_l_vec.sum()
    
def chooseSigma(prediction, counts, Q_mat_all, X_vals, sigma):
    X = prediction.values.T.reshape(-1)
    X = np.maximum(X, 1e-4)
    Y = counts.T.reshape(-1)
    num_sample = min(1000000, X.shape[0])
    use_ind = np.random.choice(np.arange(X.shape[0]), num_sample, replace = False)
    X = X[use_ind]; Y = Y[use_ind]
    mult_fac_vec = np.arange(8,13) / 10
    sigma_ind = np.concatenate((np.arange(10,71), np.arange(72, 202,2)))
    si = np.where(sigma_ind == np.around(sigma))[0].item() + 1
    sigma_ind = sigma_ind[(max(1,si - 8) - 1):(min(si+8,sigma_ind.shape[0]))]
    score_vec = np.zeros(sigma_ind.shape[0])
    for i in range(sigma_ind.shape[0]):
        sigma = sigma_ind[i]
#         set_likelihood_vars(Q_mat_all[str(sigma)],X_vals)
        likelihood_vars_sigma = {'Q_mat': Q_mat_all[str(sigma)], 'X_vals': X_vals, 'N_X': Q_mat_all[str(sigma)].shape[1], 'K_val': Q_mat_all[str(sigma)].shape[0] - 3}
        best_val = calc_log_l_vec(X*mult_fac_vec[0], Y, likelihood_vars = likelihood_vars_sigma)
        for mult_fac in mult_fac_vec[1:]:
            best_val = min(best_val, calc_log_l_vec(X*mult_fac, Y, likelihood_vars = likelihood_vars_sigma))
        score_vec[i] = best_val
    sigma = sigma_ind[np.argmin(score_vec)]
    return sigma


def get_der_fast(S, B, S_mat, gene_list, prediction, bulk_mode = False, likelihood_vars = None):
    if bulk_mode:
        d1_vec = (np.log(prediction) - np.log(B)) / prediction * (-2)
        d2_vec = ((1 - np.log(prediction) + np.log(B))/prediction**2) * (-2)
    else:
        d1_d2 = get_d1_d2(B, prediction, likelihood_vars = likelihood_vars)
        d1_vec = d1_d2['d1_vec']
        d2_vec = d1_d2['d2_vec']
    
    grad = -d1_vec @ S
    hess = (-d2_vec.to_numpy()[:,None,None] * S_mat).sum(0)
    return {'grad': grad, 'hess': hess}

def psd(H):
    eig = np.linalg.eig(H)
    epsilon = 1e-3
    if H.shape[0] == 1:
        P = eig[1] @ np.maximum(eig[0], epsilon) @ eig[1].T
#         P = eig[1] @ np.clip(eig[0], a_min = epsilon, a_max = eig[0].max() + 10) @ eig[1].T
    else:
        P = eig[1] @ np.diag(np.maximum(eig[0], epsilon)) @ eig[1].T
#         P = eig[1] @ np.diag(np.clip(eig[0], a_min = epsilon, a_max = eig[0].max() + 10)) @ eig[1].T
    return P

def solveWLS(S,B,S_mat,initialSol, nUMI, bulk_mode = False, constrain = False, likelihood_vars = None, solver = 'osqp'):
    solution = initialSol.copy()
    solution[solution < 0] = 0
    prediction = np.absolute(S @ solution)
    threshold = max(1e-4, nUMI * 1e-7)
    prediction[prediction < threshold] = threshold
    gene_list = np.array(S.index)
    derivatives = get_der_fast(S, B, S_mat, gene_list, prediction, bulk_mode = bulk_mode, likelihood_vars = likelihood_vars)
    d_vec = -derivatives['grad']
    D_mat = psd(derivatives['hess'])
    norm_factor = np.linalg.norm(D_mat, ord = 2)
    D_mat = D_mat / norm_factor
    d_vec = d_vec / norm_factor
    epsilon = 1e-7
    D_mat = D_mat + epsilon * np.identity(d_vec.shape[0])
    A = np.identity(S.shape[1])
    bzero = -solution
    alpha = 0.3
    if constrain:
        solution = solution + alpha*solve_qp(np.array(D_mat),-np.array(d_vec),-np.array(A),-np.array(bzero), np.ones(solution.shape[0]), 1 - solution.sum(), solver=solver)
    else:
        solution = solution + alpha*solve_qp(np.array(D_mat),-np.array(d_vec),-np.array(A),-np.array(bzero), solver=solver)
    return solution

def solveIRWLS_weights(S,B,nUMI, OLS=False, constrain = True, verbose = False,
                              n_iter = 50, MIN_CHANGE = .001, bulk_mode = False, solution = None, loggings = None, likelihood_vars = None, solver = 'osqp'):
    if not bulk_mode:
        K_val = likelihood_vars['K_val']
        B = np.copy(B)
        B[B > K_val] = K_val
    solution = np.ones(S.shape[1]) / S.shape[1]
    S_mat = np.einsum('ij, ik -> ijk', S, S)
    iterations = 0
    change = 1
    changes = []
    while (change > MIN_CHANGE) & (iterations < n_iter):
        new_solution = solveWLS(S,B,S_mat,solution, nUMI,constrain=constrain, bulk_mode = bulk_mode, likelihood_vars = likelihood_vars, solver = solver)
        change = np.linalg.norm(new_solution-solution, 1)
        if verbose:
            loggings.info('Change: {}'.format(change))
            loggings.info(solution)
        solution = new_solution
        iterations += 1
    return {'weights': pd.DataFrame(solution, index = S.columns), 'converged': (change <= MIN_CHANGE)}

@ray.remote
def decompose_full_ray(args):
    cell_type_profiles, nUMI, bead, constrain, OLS, MIN_CHANGE, likelihood_vars, loggings = args
    bulk_mode = False
    verbose = False
    n_iter = 50
    try:
        results = solveIRWLS_weights(cell_type_profiles,bead,nUMI,OLS = OLS, constrain = constrain,
                                   verbose = verbose, n_iter = n_iter, MIN_CHANGE = MIN_CHANGE, bulk_mode = bulk_mode, loggings = loggings, likelihood_vars = likelihood_vars, solver = 'osqp')
    except:
        results = solveIRWLS_weights(cell_type_profiles,bead,nUMI,OLS = OLS, constrain = constrain,
                                   verbose = verbose, n_iter = n_iter, MIN_CHANGE = MIN_CHANGE, bulk_mode = bulk_mode, loggings = loggings, likelihood_vars = likelihood_vars, solver = 'cvxopt')
    return results


def decompose_full(cell_type_profiles, nUMI, bead, constrain = True, OLS = False, MIN_CHANGE = 0.001, likelihood_vars = None, loggings = None, bulk_mode = False, verbose = False, n_iter = 50):
    results = solveIRWLS_weights(cell_type_profiles,bead,nUMI,OLS = OLS, constrain = constrain,
                               verbose = verbose, n_iter = n_iter, MIN_CHANGE = MIN_CHANGE, bulk_mode = bulk_mode, loggings = loggings, likelihood_vars = likelihood_vars)
    return results


def decompose_batch(nUMI, cell_type_means, beads, gene_list, constrain = True, OLS = False, max_cores = 22, MIN_CHANGE = 0.001, likelihood_vars = None, loggings = None):
    inp_args = []
    weights = []
    for i in range(beads.shape[0]):
        K_val = likelihood_vars['K_val']
        bead = beads[i,:]
        bead[bead > K_val] = K_val
        inp_args.append((cell_type_means.loc[gene_list,]*nUMI[i], nUMI[i], bead, constrain, OLS, MIN_CHANGE, likelihood_vars, loggings))
    weights = ray.get([decompose_full_ray.remote(arg) for arg in inp_args])
    return [_['weights'].values for _ in weights]



def SpatialRNA(coords, counts, nUMI = None):
    barcodes = list(set(coords.index) & set(counts.columns) & set(nUMI.index))
    if len(barcodes) == 0:
        raise ValueError('SpatialRNA: coords, counts, and nUMI do not share any barcode names. Please ensure that rownames(coords) matches colnames(counts) and names(nUMI)')
    if len(barcodes) < max(coords.shape[0], counts.shape[1], nUMI.shape[0]):
        raise ValueError('SpatialRNA: some barcodes in nUMI, coords, or counts were not mutually shared. Such barcodes were removed.')
    spatialrna_dict = {}
    spatialrna_dict['coords'] = coords
    spatialrna_dict['counts'] = counts
    spatialrna_dict['nUMI'] = nUMI
    return spatialrna_dict

def Reference(counts, cell_types, nUMI = None, n_max_cells = 10000, loggings=None):
    reference_dict = {}
    reference_dict['cell_types'] = cell_types
    reference_dict['counts'] = counts
    reference_dict['nUMI'] = nUMI
    cur_count = reference_dict['cell_types'].value_counts().max()
    if cur_count > n_max_cells:
        loggings.info('Reference: number of cells per cell type is {}, larger than maximum allowable of {}. Downsampling number of cells to: {}.'.format(cur_count, n_max_cells, n_max_cells))
    reference = create_downsampled_data(reference_dict, n_samples = n_max_cells)
    #reference = reference_dict
    return reference
            
    
    
def create_downsampled_data(reference, cell_types_keep = None, n_samples = 10000):
    cell_types_list = np.array(reference['cell_types'].iloc[:,0])
    index_keep = []
    if cell_types_keep is None:
        cell_types_keep = np.unique(cell_types_list).tolist()
    for i in range(len(cell_types_keep)):
        new_index = cell_types_list == cell_types_keep[i]
        new_index = np.where(new_index == True)[0]
        new_samples = min(n_samples, new_index.shape[0])
        choose_index = np.random.choice(new_index, new_samples, replace = False)
#         choose_index = np.arange(new_index.shape[0])
#         random.shuffle(choose_index)
#         index_keep.append(new_index[choose_index[:new_samples]])
        index_keep.append(choose_index)
    index_keep = np.concatenate(index_keep, axis = -1)
    reference['counts'] = reference['counts'].iloc[:,index_keep]
    reference['cell_types'] = reference['cell_types'].iloc[index_keep,:]
    reference['nUMI'] = reference['nUMI'].iloc[index_keep,:]
    return reference
                   
    
def get_cell_type_info(counts, cell_types, nUMI, cell_type_names = None):
    if cell_type_names is None:
        cell_type_names = np.unique(cell_types.iloc[:,0])
    n_cell_types = cell_type_names.shape[0]
    
    def get_cell_mean(cell_type):
        index = np.array(cell_types.iloc[:,0]) == cell_type
        try:
            normData = counts.loc[:, index].values / nUMI[index].values.squeeze()[None,:]
        except:
            normData = counts.loc[:, index].values / nUMI[index].values.squeeze()
        return normData.mean(1).squeeze()
    
    cell_type_means = pd.DataFrame()
    for cell_type in cell_type_names:
        cell_type_means[cell_type] = get_cell_mean(cell_type)
    cell_type_means.index = counts.index
    cell_type_means.columns = cell_type_names
    ret = {'cell_type_means': cell_type_means, 'cell_type_names': cell_type_names, 'n_cell_types': n_cell_types}
    return ret


def process_cell_type_info(reference, cell_type_names, CELL_MIN = 25, loggings = None):
    loggings.info("Begin: process_cell_type_info")
    loggings.info("process_cell_type_info: number of cells in reference: {}".format(reference['counts'].shape[1]))
    loggings.info("process_cell_type_info: number of genes in reference: {}".format(reference['counts'].shape[0]))
    cell_counts = reference['cell_types'].value_counts()
    loggings.info(cell_counts)
    if reference['cell_types'].value_counts().min() < CELL_MIN:
        loggings.info("process_cell_type_info error: need a minimum of {} cells for each cell type in the reference".format(CELL_MIN))
    cell_type_info = get_cell_type_info(reference['counts'], reference['cell_types'], reference['nUMI'], cell_type_names = cell_type_names)
    loggings.info("End: process_cell_type_info")
    return cell_type_info

def restrict_counts(puck, gene_list, UMI_thresh = 1, UMI_max = 20000):
    keep_loc = (puck['nUMI'] >= UMI_thresh) & (puck['nUMI'] <= UMI_max)
    puck['counts'] = puck['counts'].loc[gene_list, np.array(keep_loc)]
    puck['nUMI'] = puck['nUMI'][np.array(keep_loc)]
    return puck


def get_de_genes(cell_type_info, puck, fc_thresh = 1.25, expr_thresh = .00015, MIN_OBS = 3, loggings = None):
    total_gene_list = []
    epsilon = 1e-9
#     bulk_vec = pd.DataFrame(puck['counts'].sum(1))
    bulk_vec = puck['counts'].sum(1)
    gene_list = np.array(cell_type_info['cell_type_means'].index)
    index = np.array([gene.startswith('mt-') for gene in gene_list])
    if gene_list[index].shape[0] > 0:
        gene_list = gene_list[~index]
    gene_list = np.array(list(set(gene_list) & set(np.array(bulk_vec.index))))
    if gene_list.shape[0] == 0:
        raise ValueError("get_de_genes: Error: 0 common genes between SpatialRNA and Reference objects. Please check for gene list nonempty intersection.")
#     gene_list = gene_list[np.array(bulk_vec.loc[gene_list] >= MIN_OBS).squeeze()]
    gene_list = gene_list[bulk_vec[gene_list] >= MIN_OBS]
    for cell_type in cell_type_info['cell_type_names']:
        other_mean = cell_type_info['cell_type_means'].loc[gene_list, cell_type_info['cell_type_names'] != cell_type].mean(1)
        logFC = np.log(cell_type_info['cell_type_means'].loc[gene_list,cell_type] + epsilon) - np.log(other_mean + epsilon)
        type_gene_list = np.where(((logFC > fc_thresh) & (cell_type_info['cell_type_means'].loc[gene_list,cell_type] > expr_thresh)) == True)[0]
        loggings.info("get_de_genes: {} found DE genes: {}".format(cell_type, type_gene_list.shape[0]))
        total_gene_list.append(type_gene_list)
    total_gene_list = np.concatenate(total_gene_list, axis = -1)
    gene_list = np.unique(gene_list[total_gene_list])
    loggings.info("get_de_genes: total DE genes: {}".format(gene_list.shape[0]))
    return gene_list

def restrict_puck(puck, barcodes):
    puck['counts'] =  puck['counts'].loc[:, barcodes]
    puck['nUMI'] =  puck['nUMI'].loc[barcodes]
    puck['coords'] =  puck['coords'].loc[barcodes,:]
    return puck
#gene_cutoff = 0.0002, fc_cutoff = 1., gene_cutoff_reg = 0.0003, fc_cutoff_reg = 1.25
def create_RCTD(spatialRNA, reference, markers_genes=None, max_cores = 4, test_mode = False, gene_cutoff = 0.0002, fc_cutoff = 1., gene_cutoff_reg = 0.0003, fc_cutoff_reg = 1.25, UMI_min = 100, UMI_max = 20000000, UMI_min_sigma = 100, MIN_OBS = 3,
                         class_df = None, CELL_MIN_INSTANCE = 25, cell_type_names = None, MAX_MULTI_TYPES = 4, keep_reference = False, cell_type_info = None, loggings = None, hs_ST=False):
    config = {'gene_cutoff': gene_cutoff, 'fc_cutoff': fc_cutoff, 'gene_cutoff_reg': gene_cutoff_reg, 'fc_cutoff_reg': fc_cutoff_reg, 'UMI_min': UMI_min, 'UMI_min_sigma': UMI_min_sigma, 'max_cores': max_cores,
                 'N_epoch': 8, 'N_X': 50000, 'K_val': 100, 'N_fit': 1000, 'N_epoch_bulk' :30, 'MIN_CHANGE_BULK': 0.0001, 'MIN_CHANGE_REG': 0.001, 'UMI_max': UMI_max, 'MIN_OBS': MIN_OBS, 'MAX_MULTI_TYPES': MAX_MULTI_TYPES, 'device':'cuda:2'}
    if test_mode:
        config = {'gene_cutoff': .00125, 'fc_cutoff': 0.5, 'gene_cutoff_reg': 0.002, 'fc_cutoff_reg': 0.75, 'UMI_min': 1000, 'UMI_min_sigma': 300, 'max_cores': 1,
                 'N_epoch': 1, 'N_X': 50000, 'K_val': 100, 'N_fit': 50, 'N_epoch_bulk' :4, 'MIN_CHANGE_BULK': 1, 'MIN_CHANGE_REG': 0.001, 'UMI_max': 200000, 'MIN_OBS': 3, 'MAX_MULTI_TYPES': MAX_MULTI_TYPES, 'device':'cuda:2'}
    if cell_type_names is None:
        cell_type_names = np.unique(reference['cell_types'].iloc[:,0])
    if cell_type_info is None:
        cell_type_info = {'info': process_cell_type_info(reference, cell_type_names = cell_type_names, CELL_MIN = CELL_MIN_INSTANCE, loggings = loggings), 'renorm': None, 'recon':None}
    if not keep_reference:
        reference = create_downsampled_data(reference, n_samples = 5)
    puck_original = restrict_counts(spatialRNA, np.array(spatialRNA['counts'].index), UMI_thresh = config['UMI_min'], UMI_max = config['UMI_max'])
    loggings.info('create.RCTD: getting regression differentially expressed genes: ')
    if markers_genes is not None:
        gene_list_reg = markers_genes
    else:    
        if hs_ST:
            gene_list_reg = np.intersect1d(spatialRNA['counts'].index.values, reference['counts'].index.values)
            loggings.info("get_de_genes: total DE genes: {}".format(gene_list_reg.shape[0]))
        else:
            gene_list_reg = get_de_genes(cell_type_info['info'], puck_original, fc_thresh = config['fc_cutoff_reg'], expr_thresh = config['gene_cutoff_reg'], MIN_OBS = config['MIN_OBS'], loggings = loggings)
    if gene_list_reg.shape[0] == 0:
        raise ValueError("create.RCTD: Error: 0 regression differentially expressed genes found")
    loggings.info('create.RCTD: getting platform effect normalization differentially expressed genes: ')
    if markers_genes is not None:
        gene_list_bulk = markers_genes
    else:
        if hs_ST:
            gene_list_bulk = np.intersect1d(spatialRNA['counts'].index.values, reference['counts'].index.values)
            loggings.info("get_de_genes: total DE genes: {}".format(gene_list_bulk.shape[0]))
        else:
            gene_list_bulk = get_de_genes(cell_type_info['info'], puck_original, fc_thresh = config['fc_cutoff'], expr_thresh = config['gene_cutoff'], MIN_OBS = config['MIN_OBS'], loggings = loggings)
    if gene_list_bulk.shape[0] == 0:
        raise ValueError("create.RCTD: Error: 0 bulk differentially expressed genes found")
    puck = restrict_counts(puck_original, gene_list_bulk, UMI_thresh = config['UMI_min'], UMI_max = config['UMI_max'])
    puck = restrict_puck(puck, puck['counts'].columns)
    if class_df is None:
        class_df = pd.DataFrame(cell_type_info['info']['cell_type_names'], index = cell_type_info['info']['cell_type_names'], columns = ['class'])
    internal_vars = {'gene_list_reg': gene_list_reg, 'gene_list_bulk': gene_list_bulk , 'class_df': class_df, 'cell_types_assigned': False}
    RCTD = {'spatialRNA': puck, 'reconSpatialRNA' : None,'reference': reference, 'config': config, 'cell_type_info': cell_type_info, 'internal_vars': internal_vars, 'imageInfo':{'partion':None, 'features':None, 'polygon':None}}
    return RCTD




def choose_sigma_c(RCTD, Q_mat_all, X_vals_loc,loggings = None):
    puck = RCTD['reconSpatialRNA']
    MIN_UMI = RCTD['config']['UMI_min_sigma']
    sigma = 100
    sigma_vals = Q_mat_all.keys()
    N_fit = min(RCTD['config']['N_fit'],(puck['nUMI'] > MIN_UMI).sum().item())
    if N_fit == 0:
        raise ValueError('choose_sigma_c determined a N_fit of 0! This is probably due to unusually low UMI counts per bead in your dataset. Try decreasing the parameter UMI_min_sigma. It currently is {} but none of the beads had counts larger than that.'.format(MIN_UMI))
    fit_ind = np.random.choice(puck['nUMI'][puck['nUMI'] > MIN_UMI].index, N_fit, replace = False)
    beads = puck['counts'].loc[RCTD['internal_vars']['gene_list_reg'],fit_ind].values.T
    loggings.info('chooseSigma: using initial Q_mat with sigma = {}'.format(sigma/100))
    likelihood_vars = {'Q_mat': Q_mat_all[str(sigma)], 'X_vals': X_vals_loc, 'N_X': Q_mat_all[str(sigma)].shape[1], 'K_val': Q_mat_all[str(sigma)].shape[0] - 3}
    for _ in np.arange(RCTD['config']['N_epoch']):
#         set_likelihood_vars(Q_mat_all[str(sigma)], X_vals_loc, )
        likelihood_vars['Q_mat'] = Q_mat_all[str(sigma)]
        results = decompose_batch(np.array(puck['nUMI'].loc[fit_ind]).squeeze(), RCTD['cell_type_info']['renorm']['cell_type_means'], beads, RCTD['internal_vars']['gene_list_reg'], constrain = False, max_cores = RCTD['config']['max_cores'], loggings = loggings,likelihood_vars = likelihood_vars)
        weights = np.zeros((N_fit, RCTD['cell_type_info']['renorm']['n_cell_types']))
        for i in range(N_fit):
            weights[i] = results[i].squeeze()

        prediction = RCTD['cell_type_info']['renorm']['cell_type_means'].loc[RCTD['internal_vars']['gene_list_reg'],:] @ weights.T * (puck['nUMI'].loc[fit_ind]).values.squeeze()[None,:]
        print('Likelihood value: {}'.format(calc_log_l_vec(prediction.values.T.reshape(-1), beads.reshape(-1),likelihood_vars = likelihood_vars)))
        sigma_prev = sigma
        sigma = chooseSigma(prediction, beads.T, Q_mat_all, likelihood_vars['X_vals'], sigma)
        loggings.info('Sigma value: {}'.format(sigma/100))
        if sigma == sigma_prev:
            break
            
            
    RCTD['internal_vars']['sigma'] = sigma/100
    RCTD['internal_vars']['Q_mat'] = Q_mat_all[str(sigma)]
    RCTD['internal_vars']['X_vals'] = likelihood_vars['X_vals']
    return(RCTD) 

# def fitPixels(RCTD, loggings = None):
#     RCTD['internal_vars']['cell_types_assigned'] = True
#     likelihood_vars = {'Q_mat': RCTD['internal_vars']['Q_mat'], 'X_vals': RCTD['internal_vars']['X_vals'], 'N_X': RCTD['internal_vars']['Q_mat'].shape[1], 'K_val': RCTD['internal_vars']['Q_mat'].shape[0] - 3}
#     cell_type_info = RCTD['cell_type_info']['renorm']
#     beads = RCTD['spatialRNA']['counts'].loc[RCTD['internal_vars']['gene_list_reg'],:].values.T
#     results = decompose_batch(np.array(RCTD['spatialRNA']['nUMI']).squeeze(), cell_type_info['cell_type_means'], beads, RCTD['internal_vars']['gene_list_reg'], constrain = False,
#                                   max_cores = RCTD['config']['max_cores'], MIN_CHANGE = RCTD['config']['MIN_CHANGE_REG'], loggings = loggings, likelihood_vars = likelihood_vars)
#     weights = np.zeros((len(results), RCTD['cell_type_info']['renorm']['n_cell_types']))
#     for i in range(weights.shape[0]):
#         weights[i] = results[i].squeeze()

#     RCTD['results'] = pd.DataFrame(weights, index = RCTD['spatialRNA']['counts'].columns, columns = RCTD['cell_type_info']['renorm']['cell_type_names'])
#     return RCTD


def fitPixels(RCTD, doublet_mode = 'full', loggings = None):
    RCTD['internal_vars']['cell_types_assigned'] = True
    likelihood_vars = {'Q_mat': RCTD['internal_vars']['Q_mat'], 'X_vals': RCTD['internal_vars']['X_vals'], 'N_X': RCTD['internal_vars']['Q_mat'].shape[1], 'K_val': RCTD['internal_vars']['Q_mat'].shape[0] - 3}
    cell_type_info = RCTD['cell_type_info']['renorm']

    if doublet_mode == 'full':
        beads = RCTD['reconSpatialRNA']['counts'].loc[RCTD['internal_vars']['gene_list_reg'],:].values.T
        results = decompose_batch(np.array(RCTD['reconSpatialRNA']['nUMI']).squeeze(), cell_type_info['cell_type_means'], beads, RCTD['internal_vars']['gene_list_reg'], constrain = False,
                                      max_cores = RCTD['config']['max_cores'], MIN_CHANGE = RCTD['config']['MIN_CHANGE_REG'], loggings = loggings, likelihood_vars = likelihood_vars)
        weights = np.zeros((len(results), RCTD['cell_type_info']['renorm']['n_cell_types']))
        for i in range(weights.shape[0]):
            weights[i] = results[i].squeeze()

        RCTD['results'] = pd.DataFrame(weights, index = RCTD['reconSpatialRNA']['counts'].columns, columns = RCTD['cell_type_info']['renorm']['cell_type_names'])
        return RCTD
    
    
    
    
