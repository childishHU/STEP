import sys
import pandas as pd
import numpy as np
import os

from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist, pdist, squareform
import itertools
from utils_MH import *
# from qpsolvers.solvers.qpswift_ import qpswift_solve_qp
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_type = 'float32'
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def find_neighbors(position, q=0.004, p=1):
    pdist = distance_matrix(position,position,p = p)
    radius = np.quantile(pdist[pdist!=0],q)
    neighbors = (pdist <= radius) & (pdist > 0)
    return [np.where(neighbors[i] == 1)[0] for i in range(neighbors.shape[0])]

def KeepOrderUnique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def cosine_similarity(vec1, vec2, exp=True):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    if exp:
        return np.exp(similarity)
    else:
        return similarity


#Greedy Algorithm
@ray.remote
def UpdateCellLabel_Greedy(args, signature_matrix, init, df_j, df_s, P, cellWeight, spotWeight, likelihood_vars):
    Y, index, pos_celltype, nUMI, alpha, cell_num_total, nu, Spot_index = args
    init_k = init['k'].copy()
    if index.shape[0] == 1:
        score_vec = np.zeros(pos_celltype.shape[0])
        choose_from = pos_celltype
        scaler = nUMI * alpha / cell_num_total
        j_vector = df_j[index[0]]
        s_vector = df_s[Spot_index]

        for k in range(choose_from.shape[0]):
            j = choose_from[k]
            mu_hat = signature_matrix.loc[:, cell_type_names[j]].values.squeeze() * init['X_Beta'].iloc[:, index[0]].values.squeeze()
            prediction = mu_hat * scaler * P[Spot_index, index[0]]
            likelihood = calc_log_l_vec(prediction.reshape(-1), Y, likelihood_vars=likelihood_vars)
            score_vec[k] = likelihood
            if j_vector.shape[0] != 0:
                prior = ((init_k[j_vector] != j) * cellWeight[index[0], j_vector]).sum() * nu / j_vector.shape[0]
                score_vec[k] += prior
        init_k[index] = choose_from[np.argmin(score_vec)]
    else:
        scaler = alpha * nUMI / cell_num_total
        s_vector = df_s[Spot_index]
        for h in np.random.choice(np.arange(len(com[index.shape[0] - 2])), len(com[index.shape[0] - 2]), replace=False):
            cells_index_small = com[index.shape[0] - 2][h]
            cell_index = index[np.array(cells_index_small)]
            other_index = init_k[index[~np.isin(index, cell_index)]]

            temp_weight = np.zeros((1, cell_types))

            if other_index.shape[0] == 0:
                other_mu = 0
            else:
                p_temp = np.array([P[Spot_index, i] for i in index[~np.isin(index, cell_index)]])
                other_mu = (signature_matrix.loc[:, [cell_type_names[_] for _ in other_index]].values * init['X_Beta'].iloc[:, index[~np.isin(index, cell_index)]].values
                            * p_temp[None, :]).sum(-1)

                for j in range(other_index.shape[0]):
                    temp_weight[0, other_index[j]] += p_temp[j]

            ct_all = [(x, y) for x in pos_celltype for y in pos_celltype]
            j_vector1 = df_j[cell_index[0]]
            j_vector2 = df_j[cell_index[1]]
            score_vec = np.zeros(len(ct_all))

            p_temp = np.array([P[Spot_index, i] for i in cell_index])
            for p in np.random.choice(np.arange(len(ct_all)), len(ct_all), replace=False):
                ct = ct_all[p]
                mu_hat = (signature_matrix.loc[:, [cell_type_names[_] for _ in ct]].values * init['X_Beta'].iloc[:, cell_index].values
                          * p_temp[None, :]).sum(-1)
                mu_hat += other_mu
                prediction = mu_hat * scaler
                likelihood = calc_log_l_vec(prediction.reshape(-1), Y, likelihood_vars=likelihood_vars)

                if (j_vector1.shape[0] == 0) and (j_vector2.shape[0] == 0):
                    score_vec[p] = likelihood
                elif (j_vector1.shape[0] == 0) and (j_vector2.shape[0] != 0):
                    prior2 = ((init_k[j_vector2] != ct[1]) * cellWeight[cell_index[1], j_vector2]).sum() * nu / j_vector2.shape[0]
                    score_vec[p] = likelihood + prior2
                elif (j_vector1.shape[0] != 0) and (j_vector2.shape[0] == 0):
                    prior1 = ((init_k[j_vector1] != ct[0]) * cellWeight[cell_index[0], j_vector1]).sum() * nu / j_vector1.shape[0]
                    score_vec[p] = likelihood + prior1
                else:
                    Ni = 0
                    for c in pos_celltype:
                        Ni = Ni + SmoothPrior(c, ct[1], init_k, j_vector1, cell_index[1], nu, cellWeight) / (SmoothPrior(ct[1], c, init_k, j_vector2, cell_index[0], nu, cellWeight))
                    logNi = np.log(Ni)
                    prior = SmoothPrior(ct[0], ct[1], init_k, j_vector1, cell_index[1], nu, cellWeight, logp=True) - logNi
                    score_vec[p] = likelihood - prior

            init_k[cell_index] = np.array(ct_all[np.argmin(score_vec)])
        
    return index,init_k[index]


def SingleCellTypeIdentification(InitProp, spot_index_name, Q_mat_all, X_vals_loc, nu = 0, n_epoch = 8, n_neighbo=10, loggings = None, hs_ST = False):
    
    global com,cell_type_names,weights,cell_types
    
    cell_locations = InitProp['imageInfo']['cell_locations'].copy()
    if 'z' in cell_locations.columns and cell_locations['z'].dtype in [np.float64,np.float32,int]:
        df_j = find_neighbors(cell_locations.loc[:,['x', 'y', 'z']].values, q = n_neighbo/cell_locations.shape[0])
    else:
        df_j = find_neighbors(cell_locations.loc[:,['x', 'y']].values, q = n_neighbo/cell_locations.shape[0])

    
    device = torch.device(InitProp['config']['device'])
    cell_types = InitProp['cell_type_info']['renorm']['n_cell_types']
    sp_index = np.array(KeepOrderUnique(cell_locations[spot_index_name]))
    spot_locations = InitProp['reconSpatialRNA']['coords'].loc[sp_index].copy()
    sp_index_table = pd.DataFrame(np.arange(sp_index.shape[0]),index = sp_index)
    if 'z' in spot_locations.columns and spot_locations.dtype in [np.float64,np.float32,int]:
        df_s = find_neighbors(spot_locations.loc[:,['x', 'y', 'z']].values, q = n_neighbo / 2 / spot_locations.shape[0])
    else:
        df_s = find_neighbors(spot_locations.loc[:,['x', 'y']].values, q = n_neighbo / 2 / spot_locations.shape[0])
    P = InitProp['imageInfo']['partion'].loc[sp_index,cell_locations.index].values
    init = dict()
    MH = True
    if hs_ST:
        try:
            weights = InitProp['results']['weights'].loc[sp_index].values
            cell_type_names = InitProp['results']['weights'].columns.values
        except:
            weights = InitProp['results'].loc[sp_index].values
            cell_type_names = InitProp['results'].columns.values
        
        if InitProp['imageInfo']['features'] is None:
            MH = False
       
    else:
        weights = InitProp['results'].loc[sp_index].values
        cell_type_names = InitProp['results'].columns.values
        
    
    if MH:
        X = pd.DataFrame(scaler.fit_transform(InitProp['imageInfo']['features']), index = InitProp['imageInfo']['features'].index, columns=InitProp['imageInfo']['features'].columns).loc[cell_locations.index,:]
        cellWeight = np.exp(squareform(pdist(X, 'cosine')))
        init['Beta'] = pd.DataFrame(np.zeros((X.shape[1], len(InitProp['internal_vars']['gene_list_reg']))), index = X.columns, columns = InitProp['internal_vars']['gene_list_reg'])
        init['Beta'] = init['Beta'].astype(data_type)
        init['Gamma'] = pd.DataFrame(np.random.choice([0, 1], size=(X.shape[1], len(InitProp['internal_vars']['gene_list_reg'])), replace=True), index = X.columns, columns = InitProp['internal_vars']['gene_list_reg'])
        init['X_Beta'] = np.exp(X @ init['Beta']).T
        init['X_Beta'] = pd.DataFrame(restrict_X_Beta(init['X_Beta']), index=InitProp['internal_vars']['gene_list_reg'], columns=X.index)
    else:
        cellWeight = np.ones((cell_locations.shape[0], cell_locations.shape[0]))
        init['X_Beta'] = pd.DataFrame(np.ones((len(InitProp['internal_vars']['gene_list_reg']), cell_locations.shape[0])), index=InitProp['internal_vars']['gene_list_reg'], columns=cell_locations.index)
    
    alpha = weights.sum(1)
    spot_label = sp_index_table.loc[cell_locations[spot_index_name].values].values.squeeze()
    nUMI = InitProp['reconSpatialRNA']['nUMI'].loc[sp_index].values.squeeze()
    signature_matrix = InitProp['cell_type_info']['renorm']['cell_type_means'].loc[InitProp['internal_vars']['gene_list_reg'], cell_type_names]

    cell_num_total = np.array([np.where(cell_locations[spot_index_name].values == sp_name)[0].shape[0] for sp_name in sp_index])
    cell_num_total_P = P.sum(axis=1)
    
    weights_long = weights[spot_label,:]

    pos_celltype = []
    for i in range(weights.shape[0]):
        candidates = np.where(weights[i] > min(weights[i].sum() / (2 * cell_num_total.max()), weights[i].sum() / 5))[0]
        if candidates.shape[0] == 0:
            candidates = np.array([0,1,2])
        pos_celltype.append(candidates)
    

    """pos_celltype_long = []
    for i in range(weights_long.shape[0]):
        candidates = np.where(weights_long[i] > min(weights_long[i].sum() / (2 * cell_num_total.max()), weights_long[i].sum() / 5))[0]
        if candidates.shape[0] == 0:
            candidates = np.array([0,1,2])
        pos_celltype_long.append(candidates)"""
    
    com = []
    for i in np.arange(2,cell_num_total.max() + 1):
        com.append(list(itertools.combinations(range(i), 2)))
    
    init['k'] = np.argmax(weights_long, axis = -1)
    
    sigma = InitProp['internal_vars']['sigma'] * 100
    sigma = round(sigma)
    puck = InitProp['reconSpatialRNA']
    MIN_UMI = InitProp['config']['UMI_min_sigma']
    
    puck_counts = puck['counts'].loc[:, sp_index]

    spotWeight = np.exp(squareform(pdist(puck_counts.loc[InitProp['internal_vars']['gene_list_reg'],:].T.values,'cosine')))
    puck_nUMI = puck['nUMI'].loc[sp_index]

    N_fit = min(InitProp['config']['N_fit'],(puck_nUMI > MIN_UMI).sum().item())
    if N_fit == 0:
        raise ValueError('choose_sigma_c determined a N_fit of 0! This is probably due to unusually low UMI counts per bead in your dataset. Try decreasing the parameter UMI_min_sigma. It currently is {} but none of the beads had counts larger than that.'.format(MIN_UMI))
    fit_ind = np.random.choice(puck_nUMI[puck_nUMI > MIN_UMI].index, N_fit, replace = False)
    beads = puck_counts.loc[InitProp['internal_vars']['gene_list_reg'],fit_ind].values.T
    loggings.info('chooseSigma: using initial Q_mat with sigma = {}'.format(sigma/100))
    puck_nUMI = puck_nUMI * alpha[:, None]

    

    signature_matrix_ref = ray.put(signature_matrix)
    df_j_ref = ray.put(df_j)
    df_s_ref = ray.put(df_s)
    P_ref = ray.put(P)
    cellWeight_ref = ray.put(cellWeight)
    spotWeight_ref = ray.put(spotWeight)

    for epoch in range(n_epoch):
        likelihood_vars = {'Q_mat': Q_mat_all[str(sigma)], 'X_vals': X_vals_loc, 'N_X': Q_mat_all[str(sigma)].shape[1], 'K_val': Q_mat_all[str(sigma)].shape[0] - 3}        
        """for gene in np.random.choice(InitProp['internal_vars']['gene_list_reg'], 500, replace=False):
            res = run_MH_single(puck_counts.loc[gene, :].values, X.values, puck_nUMI.values,init['Beta'][gene].values, init['Gamma'][gene].values,likelihood_vars, P, signature_matrix.loc[gene,[cell_type_names[ct] for ct in init['k']]].values)
            init['Beta'][gene] = res['beta']
            init['Gamma'][gene] = res['gamma']"""
        
        likelihood_vars_ref = ray.put(likelihood_vars)
        init_ref = ray.put(init)

        print("Updating cell labels!")
        inp_args = []  
        for i in np.random.choice(np.arange(sp_index.shape[0]), sp_index.shape[0], replace = False):
            index = np.where(spot_label == i)[0]
            Y = puck_counts.loc[InitProp['internal_vars']['gene_list_reg'], sp_index[i]].values
            inp_args.append((Y,index,pos_celltype[i],nUMI[i],alpha[i],cell_num_total_P[i],nu,i))
    
            """weights[i] = 0
            for j in range(cell_num_total[i]):
                weights[i, init['k'][index][j]] += P[i, index[j]]

            weights[i] = weights[i] * alpha[i] / cell_num_total_P[i]"""

        init_update_res = ray.get([UpdateCellLabel_Greedy.remote(args, signature_matrix_ref, init_ref, df_j_ref, df_s_ref, P_ref, cellWeight_ref, spotWeight_ref, likelihood_vars_ref)
                   for args in inp_args])
        
        for init_update_index,init_update in init_update_res:
            init['k'][init_update_index] = init_update

        if MH:
            print("MCMC starts!")
            res = run_MH_full(puck_counts.loc[InitProp['internal_vars']['gene_list_reg'],:].values, X.values, puck_nUMI.values, init['Beta'].values, init['Gamma'].values, likelihood_vars, P, signature_matrix.loc[:, [cell_type_names[ct] for ct in init['k']]].values, device)
            
            init['Beta'] = pd.DataFrame(res['beta'], index = X.columns, columns = InitProp['internal_vars']['gene_list_reg'])
            init['Gamma'] = pd.DataFrame(res['gamma'], index = X.columns, columns = InitProp['internal_vars']['gene_list_reg'])
            init['X_Beta'] = np.exp(X @ init['Beta']).T
            init['X_Beta'] = pd.DataFrame(restrict_X_Beta(init['X_Beta']), index=InitProp['internal_vars']['gene_list_reg'], columns=X.index)
       
        lambda_hat = calc_lambda_hat(P, signature_matrix.loc[:, [cell_type_names[ct] for ct in init['k']]].values, init['X_Beta'].values)
        lambda_hat = pd.DataFrame(lambda_hat, index = InitProp['internal_vars']['gene_list_reg'], columns = sp_index)
        prediction = lambda_hat.loc[:, fit_ind] * (puck_nUMI.loc[fit_ind]).values.squeeze()[None,:]
        print('Likelihood value: {}'.format(calc_log_l_vec(prediction.values.T.reshape(-1), beads.reshape(-1),likelihood_vars = likelihood_vars)))
        sigma_prev = sigma
        sigma = chooseSigma(prediction, beads.T, Q_mat_all, likelihood_vars['X_vals'], sigma)
        loggings.info('Sigma value: {}'.format(sigma/100))
        if sigma_prev == sigma  and epoch > 1:
            break
    
    InitProp['internal_vars']['sigma'] = sigma/100
    if MH:
        InitProp['internal_vars']['Gamma'] = init['Gamma']
        InitProp['internal_vars']['Beta'] = init['Beta']
    InitProp['internal_vars']['Q_mat'] = Q_mat_all[str(sigma)]
    InitProp['internal_vars']['X_vals'] = likelihood_vars['X_vals']
    cell_locations['discrete_label'] = init['k']
    InitProp['discrete_label'] = cell_locations
    if hs_ST:
        try:
            InitProp['label2ct'] = pd.DataFrame(InitProp['results']['weights'].columns, index = np.arange(InitProp['results']['weights'].shape[1]))
        except:
            InitProp['label2ct'] = pd.DataFrame(InitProp['results'].columns, index = np.arange(InitProp['results'].shape[1]))
    else:
        InitProp['label2ct'] = pd.DataFrame(InitProp['results'].columns, index = np.arange(InitProp['results'].shape[1]))

    return InitProp

def SmoothPrior(i,j, init_k, j_vector, index_of_j, nu, cellWeight, logp = False):
    init_fack = init_k.copy()
    init_fack[index_of_j] = j
    U = -((init_fack[j_vector] != i) * cellWeight[index_of_j, j_vector]).sum() * nu / j_vector.shape[0]
    if logp:
        return U
    else:
        return np.exp(U)
    
