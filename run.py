from SingleLevelDecon import *
import argparse

parser = argparse.ArgumentParser(description='STEP')
parser.add_argument('--tissue', type=str, help='tissue name', default=None)
parser.add_argument('--out_dir', type=str, help='output path', default=None)
parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
parser.add_argument('--SC_Data', type=str, help='single cell reference data path', default=None)
parser.add_argument('--cell_class_column', type=str, help='input cell class label column in scRef file', default = 'cell_type')    
parser.add_argument('--InitProp', type=str, help='whether to run warmstart', default = None)   
parser.add_argument('--marker_genes', type=bool, help='whether to choose your own marker genes for selection, .var[\'highly_variable\']', default = False)   
parser.add_argument('--down_sampled', type=bool, help='whether to mask spot', default = False)   
parser.add_argument('--drop', type=bool, help='whether to drop marker genes', default = False)   
parser.add_argument('--hs_ST', type=bool, help='whether it is an image-based dataset', default=False)
parser.add_argument('--UMI_min_sigma', type=int, help='WarmStart parameter', default=300)
parser.add_argument('--n_neighbo', type=int, help='spatial prior parameter, the range of neighbor cells', default=10)
parser.add_argument('--nu', type=float, help='spatial prior parameter, higher nu means stronger spatial prior', default=10)
parser.add_argument('--model', type=str, help='GAT or Likelihood for Spatial Diffusion', default='GAT_LPA')
parser.add_argument('--searchNU', type=int, help='k in KNN in Gene Enhancement', default='50')
args = parser.parse_args()  

# python run.py --tissue TEMP --out_dir './output' --ST_Data '/home/hzq/code/idea/output/Human_Breast_Cancer/sp_adata_ef.h5ad' --SC_Data '/data115_1/hzq/idea/FFPE_Human_Breast_Cancer/sc_Human_Breast_Cancer.h5ad' --cell_class_column 'celltype_major' --searchNU 100
# Cell Type Inference & Spatial Diffusion

sl = SLDE(args.tissue, args.out_dir, args.ST_Data, args.SC_Data, cell_class_column = args.cell_class_column, InitProp = args.InitProp,
          down_sampled=args.down_sampled, marker_genes=args.marker_genes, drop=args.drop)
sl.CellTypeIdentification(nu = args.nu, n_neighbo = args.n_neighbo, hs_ST = args.hs_ST, UMI_min_sigma = args.UMI_min_sigma, VisiumCellsPlot=False, model=args.model)

# Gene Enhancement

sp_adata = anndata.read_h5ad(args.ST_Data)
del sp_adata.uns['features']['polygon']
with open(os.path.join(sl.out_dir, 'InitProp.pickle'), 'rb') as handle:
    InitProp = pickle.load(handle) 

sc_adata = sc.read_h5ad(args.SC_Data)
if sc_adata.X.max()<30:
    try:
        sc_adata.X = np.exp(sc_adata.X) - 1
    except:
        sc_adata.X = np.exp(sc_adata.X.toarray()) - 1
    sc.pp.normalize_total(sc_adata, inplace=True)
InitProp['reference']['counts'] = sc_adata.to_df().T
InitProp['reference']['cell_types'] = pd.DataFrame(sc_adata.obs[args.cell_class_column])
del sc_adata
if InitProp['reference']['cell_types'].value_counts().max() > 10000:
    final_select = []
    for cell_type in InitProp['cell_type_info']['renorm']['cell_type_names']:
        select = np.where(InitProp['reference']['cell_types']['celltype_major'] == cell_type)[0]
        if  select.shape[0] > 100:
            select = np.random.choice(select, 100, replace=False)
        final_select.append(select)
    final_select = np.concatenate(final_select)
    InitProp['reference']['counts'] = InitProp['reference']['counts'].iloc[:, final_select]
    InitProp['reference']['cell_types'] = InitProp['reference']['cell_types'].iloc[final_select]

beta = pd.read_csv(os.path.join(sl.out_dir, 'Genes_factors.csv'), index_col=0)
features = sp_adata.uns['features'].copy()
X = scaler.fit_transform(features)
x_beta = restrict_X_Beta(np.exp(X @ beta).values)

AllCT = pd.read_csv(os.path.join(sl.out_dir, 'AllCellTypeLabel_nu' + str(args.nu) + '.csv'), index_col=0)
AllCT[['x','y']] = sp_adata.uns['features'][['x','y']]
AllGENE = SearchInType(InitProp, AllCT, x_beta, sp_adata, nu=args.searchNU)
AllGENE.columns = AllCT.index

ad_all = sc.AnnData(X = AllGENE.T, obs=AllCT, var = pd.DataFrame(index=AllGENE.index))
ad_all.obsm['spatial'] = ad_all.obs[["x", "y"]].to_numpy()
ad_all.uns = sp_adata.uns
ad_all.write_h5ad(os.path.join(sl.out_dir, 'AllGENE.h5ad'))

