
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

def random_mix(Xs, k, k_min, n_samples):
    Xs_new = []
    
    fractions = np.random.rand(n_samples, k)
    num_zeros = np.random.randint(0, k - k_min + 1, size=n_samples)
    for i, nz in enumerate(num_zeros):
        fractions[i, np.random.choice(k, nz, replace=False)] = 0

    indices = np.random.randint(len(Xs), size=(n_samples, k))
    for i in range(n_samples):
        fractions[i] /= fractions[i].sum()
        mixed_sample = np.sum(Xs[indices[i]] * fractions[i][:, None], axis=0)
        Xs_new.append(mixed_sample)
    
    return np.array(Xs_new)


def load_data(adata_st, adata_sc, n_celltype, args, image_based=False, loggings=None):

    n_pseudo_scrna = int(min(100 * adata_st.shape[0] * n_celltype, args.n_samples))
    n_pseudo_spatial = n_pseudo_scrna // 2
  
    loggings.info(f'generate {n_pseudo_scrna} pseudo-spots containing {args.k_sc_min} to {args.k_sc_max} cells from scRNA-seq cells')
    minmax_sc = MinMaxScaler(feature_range=(0, args.input_max))
    mat_sc = adata_sc.copy()
    mat_sc_s = random_mix(mat_sc, args.k_sc_max, args.k_sc_min, n_pseudo_scrna).astype(np.float32)
    if not image_based:
        mat_sc_s = np.log1p(mat_sc_s)

    mat_sc = mat_sc.astype(np.float32)
    if not image_based:
        mat_sc_r = np.log1p(mat_sc)
    else:
        mat_sc_r = mat_sc
    
    mat_sc_r = minmax_sc.fit_transform(mat_sc_r)
    mat_sc_s = minmax_sc.transform(mat_sc_s)

    loggings.info(f'generate {n_pseudo_spatial} pseudo-spots containing {args.k_st_min} to {args.k_st_max} spots from spatial spots')
    minmax_st = MinMaxScaler(feature_range=(0, args.input_max))
    mat_sp = adata_st.copy()
    mat_sp_s = random_mix(mat_sp, args.k_st_max, args.k_st_min, n_pseudo_spatial).astype(np.float32)
    if not image_based:
        mat_sp_s = np.log1p(mat_sp_s)

    mat_sp = mat_sp.astype(np.float32)
    if not image_based:
        mat_sp_r = np.log1p(mat_sp)
    else:
        mat_sp_r = mat_sp

    mat_sp_r = minmax_st.fit_transform(mat_sp_r)
    mat_sp_s = minmax_st.transform(mat_sp_s)

    # training sample weights
    weight_pseudo_scrna = np.ones((mat_sc_s.shape[0],))
    weight_cell_scrna = np.ones((mat_sc_r.shape[0],))
    weight_pseudo_spatial = np.ones((mat_sp_s.shape[0],))
    weight_spot_spatial = np.ones((mat_sp_r.shape[0],))

    # weight sum of scRNA-seq cells : sum of scRNA pseudo spots = 1 : 1

    if mat_sc_s.shape[0] > 0:
        if mat_sc_s.shape[0] > mat_sc_r.shape[0]:
            weight_pseudo_scrna *= mat_sc_r.shape[0] / mat_sc_s.shape[0]
        elif mat_sc_s.shape[0] < mat_sc_r.shape[0]:
            weight_cell_scrna *= mat_sc_s.shape[0] / mat_sc_r.shape[0]
            
    # weight sum of spatial spots : sum of spatial pseudo spots = 1 : 1
    if mat_sp_s.shape[0] > 0:
        if mat_sp_s.shape[0] > mat_sp_r.shape[0]:
            weight_pseudo_spatial *= mat_sp_r.shape[0] / mat_sp_s.shape[0]
        elif mat_sp_s.shape[0] < mat_sp_r.shape[0]:
            weight_spot_spatial *= mat_sp_s.shape[0] / mat_sp_r.shape[0]

    # Final Balancing, re-weight spatial data to make sure the sum of spatial : sum of scRNA-seq = 1 : 1
    if (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna)) < (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial)):
        tmp_factor = (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna)) / (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial))
        weight_pseudo_spatial *= tmp_factor
        weight_spot_spatial *= tmp_factor
    elif (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna)) > (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial)):
        tmp_factor = (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial)) / (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna))
        weight_pseudo_scrna *= tmp_factor
        weight_cell_scrna *= tmp_factor

    sample_weight = np.concatenate([weight_pseudo_spatial, weight_spot_spatial,
                                    weight_pseudo_scrna, weight_cell_scrna])
    
    if args.use_batch_norm:
        one_batch_size = args.bs
        do_shuffle = True
    else:
        one_batch_size = data.shape[0]
        do_shuffle = False

    data = np.concatenate([mat_sp_s, mat_sp_r, mat_sc_s, mat_sc_r])
    labels = np.array([args.input_max] * (mat_sp_s.shape[0] + mat_sp_r.shape[0]) + [0.] * (mat_sc_s.shape[0] + mat_sc_r.shape[0]))
    labels = labels.reshape((len(labels), 1))
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), torch.tensor(sample_weight, dtype=torch.float32))
    loader = DataLoader(dataset, one_batch_size, do_shuffle, num_workers=8, drop_last=True)

    return data.shape[1], loader, mat_sc_r, mat_sp_r, minmax_sc