from .Identification import *
from .Extract_Features import *
import scipy.stats as st
from filelock import FileLock


def ExtractFeatures(
    tissue=None,
    out_dir=None,
    ST_Data=None,
    Img_Data=None,
    CLAM_Data=None,
    Json_Data=None,
    part=False,
    hs_ST=False
):
    """
    Parameters:
    - tissue (str): Tissue name.
    - out_dir (str): Output path.
    - ST_Data (str): Path to ST data.
    - Img_Data (str): Path to H&E stained image data.
    - CLAM_Data (str): Path to CLAM data.
    - Json_Data (str): Path to JSON data (from QuPath & StarDist).
    - part (bool): Whether all spots are distributed on tissue.
    - hs_ST (bool): Image-based or seq-based.

    Returns:
    os.path.join(os.path.join(out_dir,tissue), 'sp_adata_ef.h5ad')
    """
    process_log = os.path.join(os.path.join(out_dir,tissue), 'process.txt')
    os.makedirs(os.path.dirname(process_log), exist_ok=True)
    with FileLock(process_log + '.lock'):
        with open(process_log, 'w') as f:
            f.write(str(0))
    if not hs_ST:
        SCF = SingleCellFeatures(
            tissue, out_dir, ST_Data, Img_Data, CLAM_Data, Json_Data, hs_ST
        )
        SCF.ExtractFeatures(process_log, part)
    else:
        st_adata = sc.read_h5ad(ST_Data)
        st_adata.write_h5ad(os.path.join(os.path.join(out_dir,tissue), 'sp_adata_ef.h5ad'))

        with FileLock(process_log + '.lock'):
            with open(process_log, 'w') as f:
                f.write(str(100))

def CellIdentification(
    tissue,
    out_dir,
    ST_Data,
    SC_Data,
    cell_class_column='cell_type',
    InitProp=None,
    marker_genes=False,
    marker_list=None,
    down_sampled=False,
    drop=False,
    hs_ST=False,
    UMI_min_sigma=300,
    n_neighbo=10,
    nu=10,
    model='GAT',
    device='cuda:0',
    max_cell_number=10,
    cvae_alpha=1.0,
    cvae_beta=1.0
):
    """
    Perform Cell Type Inference and Spatial Diffusion.

    Parameters:
    - tissue (str): Tissue name.
    - out_dir (str): Output path.
    - ST_Data (str): Path to ST data (h5ad file).
    - SC_Data (str): Path to single-cell reference data (h5ad file).
    - cell_class_column (str): Column name for cell type labels in SC_Data.
    - InitProp (str or None): Path to warmstart file or None.
    - marker_genes (bool): Whether to use custom marker genes.
    - down_sampled (bool): Whether to mask spots.
    - drop (bool): Whether to drop marker genes.
    - hs_ST (bool): Whether the dataset is image-based.
    - UMI_min_sigma (int): Warmstart parameter.
    - n_neighbo (int): Spatial prior parameter (neighbor range).
    - nu (float): Spatial prior strength.
    - model (str): Model type ('GAT' or 'Likelihood').
    - device (str): 
    - max_cell_number (int):maximum cell number per spot.

    Returns:
    os.path.join(os.path.join(out_dir,tissue), f'AllCellTypeLabel_nu{InitProp["nu"]}.csv')
    os.path.join(os.path.join(out_dir,tissue), f'CellTypeLabel_nu{InitProp["nu"]}.csv')
    """
    process_log = os.path.join(os.path.join(out_dir,tissue), 'process.txt')
    with FileLock(process_log + '.lock'):
        with open(process_log, 'w') as f:
            f.write(str(0))
    sl = Identification(
        tissue=tissue,
        out_dir=out_dir,
        ST_Data=ST_Data,
        SC_Data=SC_Data,
        cell_class_column=cell_class_column,
        InitProp=InitProp,
        down_sampled=down_sampled,
        marker_genes=marker_genes,
        drop=drop,
        device=device,
        max_cell_number=max_cell_number,
        process_log=process_log,
        marker_list=marker_list,
        cvae_alpha=cvae_alpha,
        cvae_beta=cvae_beta
    )
    sl.CellTypeIdentification(
        nu=nu, 
        n_neighbo=n_neighbo, 
        hs_ST=hs_ST, 
        UMI_min_sigma=UMI_min_sigma, 
        VisiumCellsPlot=False, 
        model=model
    )
    dir = os.path.join(out_dir, tissue)
    with open(os.path.join(dir, 'InitProp.pickle'), 'rb+') as handle:
        InitProp = pickle.load(handle)
        InitProp['nu'] = nu
        handle.seek(0)
        pickle.dump(InitProp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def GeneEnhancement(
    tissue,
    out_dir,
    ST_Data,
    SC_Data,
    cell_class_column='cell_type',
    hs_cell_class_column='class_label',
    searchNU=50,
    hs_ST=False,
    device='cuda:0'
):
    """
    Run the Gene Enhancement pipeline and write results to `<out_dir>/<tissue>/AllGENE.h5ad`.

    Parameters
    ----------
    tissue : str
        Tissue identifier used for folder organization.
    out_dir : str
        Root output directory.
    ST_Data : str
        Path to the spatial transcriptomics AnnData (.h5ad) file.
    SC_Data : str
        Path to the single-cell reference AnnData (.h5ad) file.
    cell_class_column : str, optional
        Column in the single-cell metadata containing cell-type labels.
    hs_cell_class_column : str, optional
        Column containing high-level class labels (used in hs mode).
    searchNU : int, optional
        k value for the KNN search performed inside `SearchInType` / `SearchImage`.
    hs_ST : bool, optional
        If True, run the high-resolution image-based branch; otherwise use ST mode.

    Returns
    -------
    str
        Absolute path to the saved `AllGENE.h5ad` file.
    """
    process_log = os.path.join(out_dir, tissue, 'process.txt')
    with FileLock(process_log + '.lock'):
        with open(process_log, 'w') as f:
            f.write('0')

    dir_path = os.path.join(out_dir, tissue)
    with open(os.path.join(dir_path, 'InitProp.pickle'), 'rb') as handle:
        InitProp = pickle.load(handle)

    nu = InitProp.get('nu', 10)

    if not hs_ST:
        # --------------------------------------------------------------
        # ST-based enhancement branch
        # --------------------------------------------------------------
        sp_adata = anndata.read_h5ad(ST_Data)
        del sp_adata.uns['features']['polygon']  # Remove string column to simplify processing

        sc_adata = sc.read_h5ad(SC_Data)
        sc_adata.obs_names_make_unique()
        sc_adata.var_names_make_unique()
        # Back-transform log data if needed, then normalize.
        if sc_adata.X.max() < 30:
            try:
                sc_adata.X = np.exp(sc_adata.X) - 1
            except Exception:
                sc_adata.X = np.exp(sc_adata.X.toarray()) - 1
            sc.pp.normalize_total(sc_adata, inplace=True)

        InitProp['reference']['counts'] = sc_adata.to_df().T

        # Ensure the specified cell-type column exists.
        if cell_class_column not in sc_adata.obs.columns:
            raise ValueError(f"The specified column '{cell_class_column}' does not exist in the MetaData.")
        InitProp['reference']['cell_types'] = pd.DataFrame(sc_adata.obs[cell_class_column])

        del sc_adata

        if InitProp['reference']['cell_types'].value_counts().max() > 8000:
            final_select = []
            for cell_type in InitProp['cell_type_info']['renorm']['cell_type_names']:
                select_idx = np.where(InitProp['reference']['cell_types'][cell_class_column] == cell_type)[0]
                if select_idx.shape[0] > 100:
                    select_idx = np.random.choice(select_idx, 100, replace=False)
                final_select.append(select_idx)
            final_select = np.concatenate(final_select)
            InitProp['reference']['counts'] = InitProp['reference']['counts'].iloc[:, final_select]
            InitProp['reference']['cell_types'] = InitProp['reference']['cell_types'].iloc[final_select]

        # Compute morphology-derived scaling factors (requires external helpers).
        beta = pd.read_csv(os.path.join(dir_path, 'Genes_factors.csv'), index_col=0)
        features = sp_adata.uns['features'].copy()
        X = scaler.fit_transform(features)  # Uses external `scaler`
        x_beta = restrict_X_Beta(np.exp(X @ beta).values)  # Uses external `restrict_X_Beta`

        with FileLock(process_log + '.lock'):
            with open(process_log, 'w') as f:
                f.write('20')

        # Prepare cell-type labels and spatial coordinates.
        AllCT = pd.read_csv(os.path.join(dir_path, f'AllCellTypeLabel_nu{nu}.csv'), index_col=0)
        # AllCT.index can be int (e.g. HBC cell_ids) while features.index is str; the
        # index-aligned assignment below would then silently fill x/y with NaN. Cast
        # to str so the two indices match.
        AllCT.index = AllCT.index.astype(str)
        AllCT[['x', 'y']] = sp_adata.uns['features'][['x', 'y']]
        InitProp['config']['device'] = device

        # Generate enhanced gene expressions via SearchInType.
        AllGENE = SearchInType(
            InitProp,
            AllCT,
            x_beta,
            sp_adata,
            nu=searchNU,
            process_log=process_log
        )
        AllGENE.columns = AllCT.index

        # Assemble AnnData with spatial coordinates preserved.
        ad_all = sc.AnnData(X=AllGENE.T, obs=AllCT, var=pd.DataFrame(index=AllGENE.index))
        ad_all.obsm['spatial'] = ad_all.obs[['x', 'y']].to_numpy()
        ad_all.uns = sp_adata.uns

    else:
        # --------------------------------------------------------------
        # High-resolution image-based branch
        # --------------------------------------------------------------
        RNA = sc.read_h5ad(SC_Data)
        RNA_data = RNA.to_df().T.copy()

        def Log_Norm_sc(x):
            return np.log(((x / np.sum(x)) * 1_000_000) + 1)

        RNA_data = RNA_data.apply(Log_Norm_sc, axis=0)

        ST = anndata.read_h5ad(ST_Data)
        ST_data = ST.to_df().T.copy()
        cell_count = np.sum(ST_data, axis=0)

        def Log_Norm_st(x):
            return np.log(((x / np.sum(x)) * np.median(cell_count)) + 1)

        ST_data = ST_data.apply(Log_Norm_st, axis=0)

        common_genes = np.intersect1d(ST_data.index, RNA_data.index)
        train_sc = st.zscore(RNA_data.T, axis=0)[common_genes]
        train_sp = st.zscore(ST_data.T, axis=0)[common_genes]

        args = Args()
        args.device = device
        loggings = configure_logging(os.path.join(dir_path, 'logs'))
        args.k_st_max = args.k_sc_max

        loggings.info('Loading existing model, no need to train')
        cvae = torch.load(os.path.join(dir_path, 'model/model.pth'))

        p = len(common_genes)
        p_cond = 1
        latent_dim = 3 * InitProp['cell_type_info']['info']['n_cell_types']
        hidden_dim = list(np.floor(np.geomspace(latent_dim, p, args.num_hidden_layer + 2)[1:args.num_hidden_layer + 1]).astype(int))

        encoder = Encoder(p, p_cond, latent_dim, hidden_dim[::-1], use_batch_norm=args.use_batch_norm).to(args.device)
        decoder = Decoder(p, p_cond, latent_dim, hidden_dim, use_batch_norm=args.use_batch_norm).to(args.device)
        encoder.load_state_dict(cvae.encoder.state_dict())
        decoder.load_state_dict(cvae.decoder.state_dict())
        encoder.eval()
        decoder.eval()

        minmax_sc = MinMaxScaler(feature_range=(0, args.input_max))
        mat_sc_r = minmax_sc.fit_transform(train_sc.values)

        minmax_st = MinMaxScaler(feature_range=(0, args.input_max))
        mat_sp_r = minmax_st.fit_transform(train_sp.values)

        new_st, new_sc = Reconstruct(encoder, decoder, mat_sc_r, mat_sp_r, minmax_sc, args, True)
        new_st = pd.DataFrame(new_st, index=train_sp.index, columns=train_sp.columns)
        new_sc = pd.DataFrame(new_sc, index=train_sc.index, columns=train_sc.columns)

        with FileLock(process_log + '.lock'):
            with open(process_log, 'w') as f:
                f.write('20')

        cell_locations = pd.read_csv(os.path.join(dir_path, f'CellTypeLabel_nu{nu}.csv'), index_col=0)
        if hs_cell_class_column is None:
            hs_cell_class_column = cell_class_column

        Gene_pre = RNA.var_names
        pre_Genes = pd.DataFrame(
            SearchImage(
                RNA_data,
                RNA.obs,
                new_sc,
                new_st,
                Gene_pre,
                cell_locations,
                hs_cell_class_column,
                cell_class_column,
                process_log
            ),
            columns=Gene_pre
        )

        pre_Genes[ST.var_names] = ST.X
        ad_all = sc.AnnData(X=pre_Genes, obs=ST.obs)
        ad_all.obsm['spatial'] = ad_all.obs[['x', 'y']].values

    # --------------------------------------------------------------
    # Persist all results and update progress.
    # --------------------------------------------------------------
    output_path = os.path.join(dir_path, 'AllGENE.h5ad')
    ad_all.write_h5ad(output_path)

    with FileLock(process_log + '.lock'):
        with open(process_log, 'w') as f:
            f.write('100')

    return output_path