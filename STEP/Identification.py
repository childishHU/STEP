import warnings
warnings.filterwarnings('ignore')

from .utils_Identification import *
from .reconGenes import *
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
import pickle
import gzip
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import ast
import sys
import GAT_LPA
import psutil
import ray
from filelock import FileLock

data_type = 'float32'

# Ensure local package discovery.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Initialize global Ray cluster upfront.
num_cpus = psutil.cpu_count(logical=False)
os.environ["PYTHONPATH"] = os.getcwd() + ":" + os.environ.get("PYTHONPATH", "")
ray.shutdown()
ray.init(num_cpus=100, _temp_dir='/tmp/ray')


class Identification:
    """
    orchestrates warm-starting, cell-type identification, and downstream enhancements for ST datasets.
    """

    def __init__(
        self,
        tissue,
        out_dir,
        ST_Data,
        SC_Data,
        cell_class_column='cell_type',
        InitProp=None,
        down_sampled=False,
        marker_genes=False,
        drop=False,
        device='cuda:0',
        max_cell_number=10,
        process_log=None,
        marker_list=None,
        cvae_alpha=1.0,
        cvae_beta=1.0
    ):
        self.tissue = tissue
        self.out_dir = out_dir
        self.ST_Data = ST_Data
        self.SC_Data = SC_Data
        self.cell_class_column = cell_class_column
        self.down_sampled = down_sampled
        self.marker_genes = marker_genes
        self.drop = drop
        self.cvae_alpha = cvae_alpha
        self.cvae_beta = cvae_beta
        self.marker_list = marker_list
        self.device = device
        self.max_cell_number = max_cell_number

        # Prepare folders.
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, tissue), exist_ok=True)
        self.out_dir = os.path.join(out_dir, tissue)
        os.makedirs(os.path.join(self.out_dir, 'model'), exist_ok=True)

        # Logging + data.
        self.loggings = configure_logging(os.path.join(self.out_dir, 'logs'))
        self.LoadData(self.ST_Data, self.SC_Data, cell_class_column=self.cell_class_column)
        self.InitProp = InitProp

        # Progress tracking.
        self.process_log = process_log or os.path.join(self.out_dir, 'process.txt')

    @staticmethod
    def farthest_point_sampling(points, sample_size):
        """
        Classic FPS to downsample spatial coordinates.
        """
        sample_size = int(sample_size)
        print(sample_size)
        sample_indices = [0]
        distances = np.linalg.norm(points - points[0], axis=1)
        for _ in range(sample_size - 1):
            farthest_index = np.argmax(distances)
            sample_indices.append(farthest_index)
            new_distances = np.linalg.norm(points - points[farthest_index], axis=1)
            distances = np.minimum(distances, new_distances)
        return sample_indices

    def LoadData(self, ST_Data, SC_Data, cell_class_column='cell_type'):
        """
        Load ST/SC AnnData, perform log back-transform, optional marker filtering,
        and optional down-sampling.
        """
        sp_adata = anndata.read_h5ad(ST_Data)
        sc_adata = anndata.read_h5ad(SC_Data)

        sp_adata.obs_names_make_unique()
        sp_adata.var_names_make_unique()
        sc_adata.obs_names_make_unique()
        sc_adata.var_names_make_unique()

        if 'Marker' in sc_adata.var.keys():
            sel_genes = sc_adata.var.index[sc_adata.var['Marker']]
            sp_adata = sp_adata[:, sp_adata.var.index.isin(sel_genes)]

        if sp_adata.X.max() < 30:
            try:
                sp_adata.X = np.exp(sp_adata.X) - 1
            except Exception:
                sp_adata.X = np.exp(sp_adata.X.toarray()) - 1

        if sc_adata.X.max() < 30:
            try:
                sc_adata.X = np.exp(sc_adata.X) - 1
            except Exception:
                sc_adata.X = np.exp(sc_adata.X.toarray()) - 1
            sc.pp.normalize_total(sc_adata, inplace=True)

        self.sp_adata = sp_adata
        self.sc_adata = sc_adata
        self.cell_class_column = cell_class_column

        if self.down_sampled:
            sample_indices = self.farthest_point_sampling(
                self.sp_adata.obsm['spatial'],
                self.sp_adata.obsm['spatial'].shape[0] * 0.8
            )
            self.sp_adata = self.sp_adata[sample_indices, :]

    def WarmStart(self, hs_ST, UMI_min_sigma=300):
        """
        Run RCTD warm-start, save InitProp with cell locations/partition info.
        """
        self.LoadLikelihoodTable()
        counts = self.sp_adata.to_df().T

        # Determine coordinate/UMI thresholds depending on hs_ST.
        if hs_ST:
            UMI_min = 20
            if 'spatial' in self.sp_adata.obsm:
                spatial = self.sp_adata.obsm['spatial']
                if spatial.shape[1] == 2:
                    coords = pd.DataFrame(spatial, index=counts.columns, columns=['x', 'y'])
                elif spatial.shape[1] == 3:
                    coords = pd.DataFrame(spatial, index=counts.columns, columns=['x', 'y', 'z'])
                else:
                    self.loggings.error(f'Wrong spatial location shape: {spatial.shape}')
                    sys.exit()
            else:
                coords = self.sp_adata.obs[['x', 'y', 'z']] if 'z' in self.sp_adata.obs.columns else self.sp_adata.obs[['x', 'y']]
        else:
            UMI_min = 100
            spatial = self.sp_adata.obsm['spatial']
            if spatial.shape[1] == 2:
                coords = pd.DataFrame(spatial, index=counts.columns, columns=['x', 'y'])
            elif spatial.shape[1] == 3:
                coords = pd.DataFrame(spatial, index=counts.columns, columns=['x', 'y', 'z'])
            else:
                self.loggings.error(f'Wrong spatial location shape: {spatial.shape}')
                sys.exit()

        UMI_min = min(UMI_min, UMI_min_sigma)
        nUMI = pd.DataFrame(np.array(self.sp_adata.X.sum(-1)), index=self.sp_adata.obs.index)
        puck = SpatialRNA(coords, counts, nUMI)

        counts = self.sc_adata.to_df().T
        try:
            cell_types = pd.DataFrame(self.sc_adata.obs[self.cell_class_column])
        except KeyError:
            raise ValueError(f"The specified column '{self.cell_class_column}' does not exist in the MetaData.")

        nUMI = pd.DataFrame(self.sc_adata.to_df().T.sum(0))
        reference = Reference(counts, cell_types, nUMI, loggings=self.loggings)

        marker_genes = None
        if self.marker_genes:
            marker_genes = self.marker_list if self.marker_list is not None else np.intersect1d(
                self.sp_adata.var_names[self.sp_adata.var['highly_variable']],
                self.sc_adata.var_names
            )

        myRCTD = create_RCTD(
            puck,
            reference,
            max_cores=22,
            UMI_min=UMI_min,
            UMI_min_sigma=UMI_min_sigma,
            loggings=self.loggings,
            hs_ST=hs_ST,
            markers_genes=marker_genes
        )

        with FileLock(self.process_log + '.lock'):
            with open(self.process_log, 'w') as f:
                f.write(str(5))

        myRCTD['config']['device'] = self.device
        myRCTD = run_RCTD(
            myRCTD,
            self.Q_mat_all,
            self.X_vals_loc,
            self.out_dir,
            hs_ST=hs_ST,
            loggings=self.loggings,
            drop=self.drop,
            process_log=self.process_log,
            alpha=self.cvae_alpha,
            beta=self.cvae_beta
        )
        self.InitProp = myRCTD

        features = self.sp_adata.uns.get('features')

        def _extract_polygon(features_df):
            """
            Helper: parse polygon strings into python objects if present.
            """
            if features_df is None or 'polygon' not in features_df.columns:
                return None
            polygon = [ast.literal_eval(poly) for poly in features_df['polygon'].values]
            del features_df['polygon']
            return polygon

        if features is not None:
            polygon_data = _extract_polygon(features)
            if polygon_data is not None:
                self.InitProp['imageInfo']['polygon'] = polygon_data

        if hs_ST:
            cell_locations = self.sp_adata.obs.copy()
            cell_locations['spot_index'] = cell_locations.index.to_numpy()
            cell_locations['cell_index'] = cell_locations['spot_index'].map(lambda x: f"{x}_0")
            cell_locations['cell_nums'] = np.ones(len(cell_locations), dtype=int)

            valid_spots = self.InitProp['reconSpatialRNA']['counts'].columns.values
            cell_locations = cell_locations.loc[cell_locations['spot_index'].isin(valid_spots)]

            self.InitProp['imageInfo']['cell_locations'] = cell_locations
            if features is not None:
                self.InitProp['imageInfo']['features'] = features
        else:
            self.InitProp['imageInfo']['features'] = features
            spatial_key = next(iter(self.sp_adata.uns['spatial']))
            scalefactors = self.sp_adata.uns['spatial'][spatial_key]['scalefactors']
            spot_diameter = scalefactors.get(
                'actual_spot_diameter_fullres',
                scalefactors['spot_diameter_fullres']
            )

            if 'cell_locations' in self.sp_adata.uns:
                cell_locations = self.sp_adata.uns['cell_locations']
                T_spot = pd.DataFrame(self.sp_adata.obsm['spatial'], index=self.sp_adata.obs.index)
                polygon_dict = dict(zip(features.index, self.InitProp['imageInfo']['polygon']))
                partion, cell_locations = get_partition_from_cell_locations(
                    cell_locations,
                    T_spot,
                    spot_diameter,
                    polygon_dict,
                    'spot_index',
                    max_cell_number=self.max_cell_number
                )
            else:
                T_spot = pd.DataFrame(self.sp_adata.obsm['spatial'], index=self.sp_adata.obs.index)
                radius = features['Circumcircle'].values if 'Circumcircle' in features else [0] * features.shape[0]
                coord_columns = ['x', 'y', 'z'] if 'z' in features.columns else ['x', 'y']

                partion, cell_locations = get_spatial_matrix(
                    T_spot,
                    features[coord_columns],
                    spot_diameter,
                    radius,
                    self.InitProp['imageInfo']['polygon'],
                    'spot_index',
                    max_cell_number=self.max_cell_number
                )
                
            self.InitProp['imageInfo']['partion'] = partion.astype(pd.SparseDtype("float64", fill_value=0))

            self.loggings.info(f"Cells in spots:{cell_locations.shape[0]}")

            valid_spots = self.InitProp['reconSpatialRNA']['counts'].columns.values
            cell_locations = cell_locations.loc[cell_locations['spot_index'].isin(valid_spots)]
            self.InitProp['imageInfo']['cell_locations'] = cell_locations

        with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'wb') as handle:
            pickle.dump(self.InitProp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.InitProp

    def LoadLikelihoodTable(self):
        """
        Load precomputed Q matrices + X values from compressed resources.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        with gzip.open(os.path.join(base_dir, 'extdata', 'Q_mat_1_1.txt.gz'), 'rt') as f:
            lines = f.readlines()
        with gzip.open(os.path.join(base_dir, 'extdata', 'Q_mat_1_2.txt.gz'), 'rt') as f:
            lines += f.readlines()

        Q1 = {}
        for i in range(len(lines)):
            if i == 61:
                Q1[str(72)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T
            elif i == 62:
                Q1[str(74)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T
            else:
                Q1[str(i + 10)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T

        with gzip.open(os.path.join(base_dir, 'extdata', 'Q_mat_2_1.txt.gz'), 'rt') as f:
            lines2 = f.readlines()
        with gzip.open(os.path.join(base_dir, 'extdata', 'Q_mat_2_2.txt.gz'), 'rt') as f:
            lines2 += f.readlines()

        Q2 = {}
        for i in range(len(lines2)):
            Q2[str(int(i * 2 + 76))] = np.reshape(np.array(lines2[i].split(' ')).astype(np.float64), (2536, 103)).T

        Q_mat_all = Q1 | Q2

        with open(os.path.join(base_dir, 'extdata', 'X_vals.txt')) as f:
            lines_X = f.readlines()

        X_vals_loc = np.array([float(lines_X_item.strip()) for lines_X_item in lines_X])

        self.Q_mat_all = Q_mat_all
        self.X_vals_loc = X_vals_loc

    def CellTypeIdentification(
        self,
        nu=10,
        n_neighbo=10,
        hs_ST=False,
        VisiumCellsPlot=True,
        UMI_min_sigma=300,
        model='LPA_Likelihood'
    ):
        """
        End-to-end pipeline: warm-start (if needed), discrete label inference,
        optional visualization, and filling labels for all potential cells.
        """
        if self.InitProp is not None:
            self.WarmStart(hs_ST=hs_ST, UMI_min_sigma=UMI_min_sigma)
        elif not os.path.exists(os.path.join(self.out_dir, 'InitProp.pickle')):
            self.WarmStart(hs_ST=hs_ST, UMI_min_sigma=UMI_min_sigma)
        else:
            self.loggings.info('Loading existing InitProp, no need to warmstart')
            with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'rb') as handle:
                self.InitProp = pickle.load(handle)
                self.LoadLikelihoodTable()
                self.InitProp['config']['device'] = self.device

        with FileLock(self.process_log + '.lock'):
            with open(self.process_log, 'w') as f:
                f.write(str(30))

        CellTypeLabel = SingleCellTypeIdentification(
            self.InitProp,
            'spot_index',
            self.Q_mat_all,
            self.X_vals_loc,
            nu=nu,
            n_epoch=8,
            n_neighbo=n_neighbo,
            loggings=self.loggings,
            hs_ST=hs_ST,
            process_log=self.process_log
        )

        label2ct = CellTypeLabel['label2ct']
        discrete_label = CellTypeLabel['discrete_label'].copy()
        discrete_label['discrete_label_ct'] = label2ct.iloc[discrete_label['discrete_label']].values.squeeze()
        discrete_label.to_csv(os.path.join(self.out_dir, f'CellTypeLabel_nu{nu}.csv'))

        self.sp_adata.uns['cell_locations'] = discrete_label

        if 'Beta' in CellTypeLabel['internal_vars']:
            CellTypeLabel['internal_vars']['Beta'].to_csv(os.path.join(self.out_dir, 'Genes_factors.csv'))

        if hs_ST or not VisiumCellsPlot:
            fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
            sns.scatterplot(
                data=discrete_label,
                x="x",
                y="y",
                s=20,
                hue='discrete_label_ct',
                palette='tab20',
                legend=True
            )
            if not hs_ST:
                plt.gca().invert_yaxis()
            plt.axis('off')
            plt.legend(bbox_to_anchor=(0.97, .98), framealpha=0)
            plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label.png'))
            plt.close()
        elif VisiumCellsPlot:
            if self.sp_adata.obsm['spatial'].shape[1] == 2:
                fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=200)
                PlotVisiumCells(
                    self.sp_adata,
                    "discrete_label_ct",
                    size=0.4,
                    alpha_img=0.4,
                    lw=0.4,
                    palette='tab20',
                    ax=ax
                )
                plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label.png'))
                plt.close()

        with FileLock(self.process_log + '.lock'):
            with open(self.process_log, 'w') as f:
                f.write(str(80))

        if not hs_ST:
            features_uns = self.sp_adata.uns['features']
            if 'polygon' in features_uns.columns:
                del features_uns['polygon']

            features = features_uns.copy()
            n_classes = self.InitProp['cell_type_info']['renorm']['n_cell_types']
            idx_train = features.index.get_indexer(discrete_label.index)

            labels = np.random.randint(0, n_classes, size=len(features))
            labels[idx_train] = discrete_label['discrete_label'].to_numpy()
            self.loggings.info("Filling Up!")

            coord_cols = ['x', 'y', 'z'] if 'z' in features.columns else ['x', 'y']
            shared_kwargs = dict(
                epochs=100,
                features=features.values,
                cell_locations=features[coord_cols],
                labels=labels,
                idx_train=idx_train,
                nclass=n_classes,
                n_neighbo=n_neighbo,
            )

            if model == 'LPA_Likelihood':
                all_label = GAT_LPA.LPA_likelihood(**shared_kwargs)
            else:
                all_pred, gat_model = GAT_LPA.train_GAT_LPA(
                    label2ct=label2ct,
                    device=self.InitProp['config']['device'],
                    **shared_kwargs,
                )
                torch.save(gat_model, os.path.join(self.out_dir, 'model/GAT.pth'))
                all_label = np.argmax(all_pred, axis=-1)
                all_label[idx_train] = labels[idx_train]

            features['discrete_label_ct'] = label2ct.iloc[all_label].values.squeeze()

            fig, ax = plt.subplots(figsize=(10, 8.5), dpi=100)
            sns.scatterplot(
                data=features,
                x='x',
                y='y',
                s=5,
                hue='discrete_label_ct',
                palette='tab20',
                legend=True,
            )
            plt.axis('off')
            plt.gca().invert_yaxis()
            plt.legend(bbox_to_anchor=(0.97, 0.98), framealpha=0)
            plt.savefig(os.path.join(self.out_dir, 'estemated_all_ct_label.png'))
            plt.close()

            features['discrete_label_ct'].to_csv(
                os.path.join(self.out_dir, f'AllCellTypeLabel_nu{nu}.csv')
            )

        with FileLock(self.process_log + '.lock'):
            with open(self.process_log, 'w') as f:
                f.write(str(100))

        ray.shutdown()