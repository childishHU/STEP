import os
import anndata
import scanpy as sc
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from STEP.Identification import *
from STEP.GAT_LPA.FillUp import LETSTransfer
from scipy.io import mmread
scaler = StandardScaler()

# Parameters
train = ['06', '11', '18', '24', '27', '32']
groups = [
    ['01', '02', '03', '04', '05', '07'],
    ['08', '09', '10', '12', '13'],
    ['14', '15', '16', '17', '19', '20'],
    ['21', '22', '23'],
    ['25', '26', '28', '29'],
    ['30', '31', '33', '34', '35']
]
model_params = {
    'hidden': 128,
    'dropout': 0.2,
    'gatnum': 2,
    'lr': 0.01,
    'Lambda': 1,
    'seed': 20230825,
    'lpaiters': 5,
    'gat_heads': 2
}
LPA_epoch = 10
out_dir = '/data/hzq/idea/Mouse_brain_3D/output'

# Cell Types
print('Transfer!')

cell2table = pd.read_csv('/data/hzq/code/allen/cellstable.csv', index_col=0)
one_hot_matrix = pd.get_dummies(cell2table['CT'])
cell2table['new_z'] = cell2table['AP'] - 530 / 2 
cell2table['new_x'] = -cell2table['DV'] * 1000 / 25 - 320 / 2
cell2table['new_y'] = cell2table['ML'] * 1000 / 25 
celltodigtal = pd.DataFrame(np.arange(len(np.unique(cell2table['CT']))), index=np.unique(cell2table['CT']))

for tissue in train:
    sel_index = cell2table.index.str.startswith(tissue)
    res_labels_lpa = one_hot_matrix.loc[sel_index].copy()
    adj = csr_matrix(constructNetworkWithinSlice(cell2table.loc[sel_index]))
    for i in tqdm(range(LPA_epoch)):
        new_labels_lpa = adj @ res_labels_lpa
        new_labels_lpa = new_labels_lpa / new_labels_lpa.sum(axis=1)[:,None]
        res_labels_lpa = new_labels_lpa.copy()
    res = np.argmax(res_labels_lpa, axis=1)
    cell2table.loc[sel_index,'CT'] = one_hot_matrix.columns[res]


cell2feature = pd.DataFrame()
for i in range(1,36):
    x = str(i) if i >= 10 else '0' + str(i)
    st = sc.read_h5ad(os.path.join(out_dir, x, 'sp_adata_ef.h5ad'))
    cell2feature_temp = pd.DataFrame(st.uns['features'].values,index=[x + 'A' + j for j in st.uns['features'].index], columns=st.uns['features'].columns)
    cell2feature = pd.concat([cell2feature, cell2feature_temp])

cell2feature = cell2feature.loc[cell2table.index]
del cell2feature['polygon']
cell2feature = cell2feature.astype(np.float32)
cell2feature = cell2feature.apply(lambda x: x.fillna(x.mean()), axis=0)

# Run Transfer
def run_LETSTransfer(base, adj, features, cell, celltodigtal, model_params):
    labels = celltodigtal.loc[cell.loc[:, 'CT']].values.reshape(-1)
    target_cells = np.where(features.index.str.startswith(base))[0]
    output, _ = LETSTransfer(features, adj, labels, target_cells, celltodigtal.shape[0], parameters=model_params)
    return output

for group, tissue in enumerate(train):
    AllGroup = sorted(groups[group] + [tissue])
    group_cell = cell2table[cell2table.index.str.startswith(tuple(AllGroup))]
    group_features = cell2feature[cell2table.index.str.startswith(tuple(AllGroup))]
    # group_adj = mmread(f'/data/hzq/idea/Mouse_brain_3D/transfer/group{group + 1}/group{group + 1}.mtx').tocsr()
    group_adj = constructFullNetwork(cell2table, AllGroup)
    group_output = run_LETSTransfer(tissue, group_adj, group_features, group_cell, celltodigtal, model_params)
    cell2table.loc[group_features.index, 'new_CT'] = np.unique(cell2table['CT'])[group_output.argmax(axis=1)]

cell2table.to_csv('/data/hzq/code/allen/celltransfer.csv')


# Genes
print('Transcriptomic Enhancement!')

searchNU = 10
nu = 10
celltable = pd.read_csv('/data/hzq/code/allen/celltransfer.csv',index_col=0)
celltable = celltable.rename(columns={'new_CT':'discrete_label_ct'})
# Download from https://www.molecularatlas.org/data-to-download/intermediary_data/figures.zip
spotable = pd.read_csv('/data/hzq/code/allen/spotstable.csv', index_col=0)
SC_Data = '/data/hzq/idea/Mouse_brain_3D/E-MTAB-11115/sc.h5ad'
sc_adata = sc.read_h5ad(SC_Data)
if sc_adata.X.max() < 30:
    try:
        sc_adata.X = np.exp(sc_adata.X) - 1
    except:
        sc_adata.X = np.exp(sc_adata.X.toarray()) - 1
    sc.pp.normalize_total(sc_adata, inplace=True)

for group, tissue in enumerate(train):
    cell_class_column = 'annotation_1'
    dir = os.path.join(out_dir, tissue)
    
    ST_Data = os.path.join(dir, 'sp_adata_ef.h5ad')

    with open(os.path.join(dir, 'InitProp.pickle'), 'rb') as handle:
        InitProp = pickle.load(handle)
    
    
    sp_adata_ref = anndata.read_h5ad(ST_Data)
    del sp_adata_ref.uns['features']['polygon']
    sp_adata_ref.obsm['spatial'] = spotable.loc[sp_adata_ref.obs_names,['ML', 'DV']].values


    InitProp['reference']['counts'] = sc_adata.to_df().T

    try:
        InitProp['reference']['cell_types'] = pd.DataFrame(sc_adata.obs[cell_class_column])
    except KeyError:
        raise ValueError(f"The specified column '{cell_class_column}' does not exist in the MetaData.")

    if InitProp['reference']['cell_types'].value_counts().max() > 10000:
        final_select = []
        for cell_type in InitProp['cell_type_info']['renorm']['cell_type_names']:
            select = np.where(InitProp['reference']['cell_types'][cell_class_column] == cell_type)[0]
            if select.shape[0] > 100:
                select = np.random.choice(select, 100, replace=False)
            final_select.append(select)
        final_select = np.concatenate(final_select)
        InitProp['reference']['counts'] = InitProp['reference']['counts'].iloc[:, final_select]
        InitProp['reference']['cell_types'] = InitProp['reference']['cell_types'].iloc[final_select]

    for tissue_transfer in groups[group]:
        print(tissue_transfer)
        dir_transfer = os.path.join(out_dir, tissue_transfer)
        ST_Data_transfer = os.path.join(dir_transfer, 'sp_adata_ef.h5ad')
        # Load spatial transcriptomics data
        sp_adata = anndata.read_h5ad(ST_Data_transfer)
        del sp_adata.uns['features']['polygon']
        sp_adata.uns['features'].index = [tissue_transfer + 'A' + i for i in sp_adata.uns['features'].index]

        # Calculate gene factors
        AllCT = celltable.loc[celltable.index.str.startswith(tissue_transfer)]
        AllCT[['x', 'y']] = AllCT[['ML','DV']]
        beta = pd.read_csv(os.path.join(dir, 'Genes_factors.csv'), index_col=0)
        features = sp_adata.uns['features'].loc[AllCT.index].copy()
        beta = beta.loc[features.columns]
        X = scaler.fit_transform(features)
        x_beta = restrict_X_Beta(np.exp(X @ beta).values)

        InitProp['config']['device'] = 'cuda:2'
        AllGENE = SearchInType(InitProp, AllCT, x_beta, sp_adata_ref, nu=searchNU)
        AllGENE.columns = AllCT.index

        ad_all = sc.AnnData(X=AllGENE.T, obs=AllCT, var=pd.DataFrame(index=AllGENE.index))
        ad_all.obsm['spatial'] = features[['x','y']].to_numpy()
        ad_all.uns = sp_adata.uns

        ad_all.write_h5ad(os.path.join(dir_transfer, 'AllGeneTran.h5ad'))


# Proteins
print('Omic Enhancement!')
inCTIE_seq = sc.read_h5ad('/data/hzq/idea/Mouse_brain_3D/GSE163480/inCTIE_seq.h5ad')

for i in range(1, 36):
    x =  '0' + str(i) if i < 10 else str(i)
    select_index = [i for i in celltable.index if i.startswith(x + 'A')]

    # This inCITE-seq dataset is for the hippocampus region, so we only predict the protein expression of all cells in the hippocampus region
    select_index = np.intersect1d(select_index, celltable.index[(celltable['label'] == 'Hippocampal region')])
    if len(select_index) > 0:
        print(f"{x} slice!")
        file_path_tran = f"/data/hzq/idea/Mouse_brain_3D/output/{x}/AllGeneTran.h5ad"
        file_path_gene = f"/data/hzq/idea/Mouse_brain_3D/output/{x}/AllGENE.h5ad"
        if os.path.exists(file_path_tran):
            allgene = sc.read_h5ad(file_path_tran)
        else:
            allgene = sc.read_h5ad(file_path_gene)
            allgene.obs_names = [x + 'A' + i for i in allgene.obs_names]

        allgene = allgene[select_index,:]
        allprotein_df = searchProtein(inCTIE_seq, allgene, ['p65_norm', 'c-Fos_norm', 'NeuN_norm', 'PU.1_norm'])
        allprotein = sc.AnnData(X=allprotein_df)
        allprotein.obs = allgene.obs
        allprotein.uns = allgene.uns
        allprotein.obsm = allgene.obsm
        allprotein.write_h5ad(f"/data/hzq/idea/Mouse_brain_3D/output/{x}/AllPROTEIN_train.h5ad")