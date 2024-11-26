import warnings
warnings.filterwarnings('ignore')
from utils_SLDE import *
from reconGenes import *
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
import pickle
import gzip
data_type = 'float32'

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import copy
import ast

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import GAT_LPA
num_cpus = psutil.cpu_count(logical=False) 
os.environ["PYTHONPATH"] = os.getcwd()+ ":" + os.environ.get("PYTHONPATH", "")
ray.shutdown()
ray.init(num_cpus=num_cpus-2,_temp_dir='/tmp/ray')



class SLDE:
    def __init__(self,tissue,out_dir, ST_Data, SC_Data, cell_class_column = 'cell_type', InitProp = None, down_sampled=False, marker_genes=False, drop=False):
        self.tissue = tissue
        self.out_dir = out_dir 
        self.ST_Data = ST_Data
        self.SC_Data = SC_Data
        self.cell_class_column = cell_class_column
        self.down_sampled = down_sampled
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(os.path.join(out_dir,tissue)):
            os.mkdir(os.path.join(out_dir,tissue))
        
        self.out_dir = os.path.join(out_dir,tissue)
        if not os.path.exists(os.path.join(self.out_dir,'model')):
            os.mkdir(os.path.join(self.out_dir,'model'))
        loggings = configure_logging(os.path.join(self.out_dir,'logs'))
        self.loggings = loggings 
        self.LoadData(self.ST_Data, self.SC_Data, cell_class_column = self.cell_class_column)
        self.InitProp = InitProp
        self.marker_genes = marker_genes
        self.drop = drop
        
    
    @staticmethod
    def farthest_point_sampling(points, sample_size):
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

        
        
    def LoadData(self, ST_Data, SC_Data, cell_class_column = 'cell_type'):
        sp_adata = anndata.read_h5ad(ST_Data)
        sc_adata = anndata.read_h5ad(SC_Data)
        sp_adata.obs_names_make_unique()
        sp_adata.var_names_make_unique()
        sc_adata.obs_names_make_unique()
        sc_adata.var_names_make_unique()

        if 'Marker' in sc_adata.var.keys():
            sel_genes = sc_adata.var.index[sc_adata.var['Marker']]
            sp_adata = sp_adata[:,sp_adata.var.index.isin(sel_genes)]

        # cell_class_column = args.cell_class_column

        if sp_adata.X.max()<30:
            try:
                sp_adata.X = np.exp(sp_adata.X) - 1
            except:
                sp_adata.X = np.exp(sp_adata.X.toarray()) - 1
        
        if sc_adata.X.max()<30:
            try:
                sc_adata.X = np.exp(sc_adata.X) - 1
            except:
                sc_adata.X = np.exp(sc_adata.X.toarray()) - 1
            sc.pp.normalize_total(sc_adata, inplace=True)
            
        self.sp_adata = sp_adata
        self.sc_adata = sc_adata
        self.cell_class_column = cell_class_column
        if self.down_sampled:
            sample_indices = self.farthest_point_sampling(self.sp_adata.obsm['spatial'], self.sp_adata.obsm['spatial'].shape[0] * 0.8)
            self.sp_adata = self.sp_adata[sample_indices, :]
        
        
    def WarmStart(self,hs_ST,UMI_min_sigma = 300):
        self.LoadLikelihoodTable()
        counts = self.sp_adata.to_df().T
        if hs_ST:
            UMI_min = 20
            if 'z' in self.sp_adata.obs.columns:
                coords = self.sp_adata.obs[['x', 'y', 'z']]
            else:
                coords = self.sp_adata.obs[['x', 'y']]
            
        else:
            UMI_min =100
            if self.sp_adata.obsm['spatial'].shape[1] == 2:
                coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y'])
                self.loggings.info('A single ST data with spatial location shape: {}'.format(self.sp_adata.obsm['spatial'].shape))
            elif self.sp_adata.obsm['spatial'].shape[1] == 3:
                coords = pd.DataFrame(self.sp_adata.obsm['spatial'], index = counts.columns, columns = ['x', 'y', 'z'])
                self.loggings.info('3D aligned ST data with spatial location shape: {}'.format(self.sp_adata.obsm['spatial'].shape))
            else:
                self.loggings.error('Wrong spatial location shape: {}'.format(self.sp_adata.obsm['spatial'].shape))
                sys.exit()
        UMI_min = min(UMI_min,UMI_min_sigma)
        nUMI = pd.DataFrame(np.array(self.sp_adata.X.sum(-1)), index = self.sp_adata.obs.index)
        puck = SpatialRNA(coords, counts, nUMI)
        counts = self.sc_adata.to_df().T
        cell_types = pd.DataFrame(self.sc_adata.obs[self.cell_class_column])
        nUMI = pd.DataFrame(self.sc_adata.to_df().T.sum(0))
        reference = Reference(counts, cell_types, nUMI, loggings=self.loggings)
        marker_genes = None
        if self.marker_genes:
            marker_genes = np.intersect1d(self.sp_adata.var_names[self.sp_adata.var['highly_variable']], self.sc_adata.var_names)
        myRCTD = create_RCTD(puck, reference, max_cores = 22, UMI_min=UMI_min, UMI_min_sigma = UMI_min_sigma, loggings = self.loggings, hs_ST=hs_ST,markers_genes=marker_genes)
        myRCTD = run_RCTD(myRCTD, self.Q_mat_all, self.X_vals_loc, self.out_dir,hs_ST=hs_ST,loggings = self.loggings, drop=self.drop)
        self.InitProp = myRCTD
        if hs_ST:
            if 'features' in self.sp_adata.uns.keys():
                self.InitProp['imageInfo']['features'] = self.sp_adata.uns['features']
            cell_locations = self.sp_adata.obs.copy()
            cell_locations['spot_index'] = np.array(cell_locations.index)
            cell_locations['cell_index'] = cell_locations['spot_index'].map(lambda x:x+'_0')
            cell_locations['cell_nums'] = np.ones(cell_locations.shape[0]).astype(int)
            cell_locations = cell_locations.loc[cell_locations['spot_index'].isin(self.InitProp['reconSpatialRNA']['counts'].columns.values)]
            self.InitProp['imageInfo']['cell_locations'] = cell_locations
            self.InitProp['imageInfo']['partion'] = pd.DataFrame(np.eye(cell_locations.shape[0]), index=cell_locations.index, columns=cell_locations.index)
        else:
            self.InitProp['imageInfo']['polygon'] = [ast.literal_eval(self.sp_adata.uns['features']['polygon'].values[i]) for i in range(self.sp_adata.uns['features'].shape[0])]
            del self.sp_adata.uns['features']['polygon']
            self.InitProp['imageInfo']['features'] = self.sp_adata.uns['features']
            
            """self.InitProp['imageInfo']['partion'] = pd.DataFrame(np.ones((self.InitProp['spatialRNA']['counts'].shape[1], self.sp_adata.uns['features'].shape[0])),
                                                                    index = self.InitProp['spatialRNA']['counts'].columns,
                                                                    columns = self.sp_adata.uns['features'].index)"""
            
            T_spot = pd.DataFrame(self.sp_adata.obsm['spatial'],index=self.sp_adata.obs.index)
            spot_diameter = self.sp_adata.uns['spatial'][list(self.sp_adata.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']
            if 'z' in self.sp_adata.uns['features'].columns:
                partion, cell_locations = get_spatial_matrix(T_spot ,self.sp_adata.uns['features'][['x','y','z']], spot_diameter, self.sp_adata.uns['features']['Circumcircle'].values,self.InitProp['imageInfo']['polygon'],'spot_index') 
            else:
                partion, cell_locations = get_spatial_matrix(T_spot ,self.sp_adata.uns['features'][['x','y']], spot_diameter, self.sp_adata.uns['features']['Circumcircle'].values,self.InitProp['imageInfo']['polygon'],'spot_index') 
            self.InitProp['imageInfo']['partion'] = partion
            #self.InitProp['imageInfo']['partion'] = pd.DataFrame(np.where(self.InitProp['imageInfo']['partion']>0,1,0), index = self.InitProp['imageInfo']['partion'].index, columns=self.InitProp['imageInfo']['partion'].columns)
            self.loggings.info('Cells in spots:{}'.format(cell_locations.shape[0]))
            cell_locations = cell_locations.loc[cell_locations['spot_index'].isin(self.InitProp['reconSpatialRNA']['counts'].columns.values)]
            self.InitProp['imageInfo']['cell_locations'] = cell_locations

        import pickle
        with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'wb') as handle:
            pickle.dump(self.InitProp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.InitProp
        
    def LoadLikelihoodTable(self):
        with gzip.open('./extdata/Q_mat_1_1.txt.gz','rt') as f:
            lines = f.readlines()
        with gzip.open('./extdata/Q_mat_1_2.txt.gz','rt') as f:
            lines += f.readlines()

        Q1 = {}
        for i in range(len(lines)):
            if i == 61:
                Q1[str(72)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T
            elif i == 62:
                Q1[str(74)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T
            else:
                Q1[str(i + 10)] = np.reshape(np.array(lines[i].split(' ')).astype(np.float64), (2536, 103)).T

        with gzip.open('./extdata/Q_mat_2_1.txt.gz','rt') as f:
            lines2 = f.readlines()
        with gzip.open('./extdata/Q_mat_2_2.txt.gz','rt') as f:
            lines2 += f.readlines()

        Q2 = {}
        for i in range(len(lines2)):
            Q2[str(int(i * 2 + 76))] = np.reshape(np.array(lines2[i].split(' ')).astype(np.float64), (2536, 103)).T

        Q_mat_all = Q1 | Q2

        with open('./extdata/X_vals.txt') as f:
            lines_X = f.readlines()

        X_vals_loc = np.array([float(lines_X_item.strip()) for lines_X_item in lines_X])
        
        self.Q_mat_all = Q_mat_all
        self.X_vals_loc = X_vals_loc
        
    def CellTypeIdentification(self, nu = 10, n_neighbo = 10, hs_ST = False, VisiumCellsPlot = True, UMI_min_sigma = 300, model='LPA_Likelihood'):
        if self.InitProp is not None:
            self.WarmStart(hs_ST=hs_ST, UMI_min_sigma = UMI_min_sigma)
        elif not os.path.exists(os.path.join(self.out_dir, 'InitProp.pickle')):
            self.WarmStart(hs_ST=hs_ST, UMI_min_sigma = UMI_min_sigma)
        elif os.path.exists(os.path.join(self.out_dir, 'InitProp.pickle')):
            self.loggings.info('Loading existing InitProp, no need to warmstart')
            with open(os.path.join(self.out_dir, 'InitProp.pickle'), 'rb') as handle:
                self.InitProp = pickle.load(handle)
                self.LoadLikelihoodTable()

        
        CellTypeLabel = SingleCellTypeIdentification(self.InitProp, 'spot_index', self.Q_mat_all, self.X_vals_loc, nu = nu, n_epoch = 8, n_neighbo = n_neighbo, loggings = self.loggings, hs_ST = hs_ST)
        label2ct = CellTypeLabel['label2ct']
        discrete_label = CellTypeLabel['discrete_label'].copy()
        discrete_label['discrete_label_ct'] = label2ct.iloc[discrete_label['discrete_label']].values.squeeze()
        discrete_label.to_csv(os.path.join(self.out_dir, 'CellTypeLabel_nu' + str(nu) + '.csv'))
        self.sp_adata.uns['cell_locations'] = discrete_label
        if 'Beta' in CellTypeLabel['internal_vars'].keys():
            CellTypeLabel['internal_vars']['Beta'].to_csv(os.path.join(self.out_dir, 'Genes_factors.csv'))
        if hs_ST or not VisiumCellsPlot:
            fig, ax = plt.subplots(figsize=(15,15),dpi=100)
            sns.scatterplot(data=discrete_label, x="x",y="y",s=20,hue='discrete_label_ct',palette='tab20',legend=True)
            if not hs_ST:
                plt.gca().invert_yaxis()
            plt.axis('off')
            plt.legend(bbox_to_anchor=(0.97, .98),framealpha=0)
            plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label.png'))
            plt.close()
        
        elif VisiumCellsPlot:
            if self.sp_adata.obsm['spatial'].shape[1] == 2:
                fig, ax = plt.subplots(1,1,figsize=(14, 8),dpi=200)
                PlotVisiumCells(self.sp_adata,"discrete_label_ct",size=0.4,alpha_img=0.4,lw=0.4,palette='tab20',ax=ax)
                plt.savefig(os.path.join(self.out_dir, 'estemated_ct_label.png'))
                plt.close()

        if not hs_ST:
            if 'polygon' in self.sp_adata.uns['features'].columns:
                del self.sp_adata.uns['features']['polygon']
            features = self.sp_adata.uns['features'].copy()
            idx_train = [dict(zip(features.index, range(features.shape[0])))[i] for i in discrete_label.index]
            labels = np.random.randint(0,self.InitProp['cell_type_info']['renorm']['n_cell_types'],size=features.shape[0])
            labels[idx_train] = discrete_label['discrete_label'].values
            print("Filling Up!")
            if 'z' in features.columns:
                if model == 'LPA_Likelihood':
                    all_label = GAT_LPA.LPA_likelihood(epochs=100, features=features.values, cell_locations=features[['x','y','z']]
                                , labels=labels, idx_train=idx_train 
                                , nclass=self.InitProp['cell_type_info']['renorm']['n_cell_types'], n_neighbo=n_neighbo)
                else:
                    all_label, model = GAT_LPA.train_GAT_LPA(epochs=100, features=features.values, label2ct=label2ct,cell_locations=features[['x','y','z']]
                                , labels=labels, idx_train=idx_train 
                                , nclass=self.InitProp['cell_type_info']['renorm']['n_cell_types'], n_neighbo=n_neighbo, device=self.InitProp['config']['device'])  
                    torch.save(model, os.path.join(self.out_dir, 'model/GAT_LPA.pth'))
                    all_label = np.argmax(all_label, axis=-1)
                    all_label[idx_train] = labels[idx_train]
            else:
                if model == 'LPA_Likelihood':
                    all_label = GAT_LPA.LPA_likelihood(epochs=100, features=features.values, cell_locations=features[['x','y']]
                                , labels=labels, idx_train=idx_train 
                                , nclass=self.InitProp['cell_type_info']['renorm']['n_cell_types'], n_neighbo=n_neighbo)
                else:
                    all_label, model = GAT_LPA.train_GAT_LPA(epochs=100, features=features.values, label2ct=label2ct, cell_locations=features[['x','y']]
                                , labels=labels, idx_train=idx_train 
                                , nclass=self.InitProp['cell_type_info']['renorm']['n_cell_types'], n_neighbo=n_neighbo, device=self.InitProp['config']['device']) 
                    torch.save(model, os.path.join(self.out_dir, 'model/GAT_LPA.pth'))
                    all_label = np.argmax(all_label, axis=-1)
                    all_label[idx_train] = labels[idx_train]
            features['discrete_label_ct'] = label2ct.iloc[all_label].values.squeeze()
            fig, ax = plt.subplots(figsize=(10,8.5),dpi=100)
            sns.scatterplot(data=features, x="x",y="y",s=5,hue='discrete_label_ct',palette='tab20',legend=True)
            plt.axis('off')
            plt.gca().invert_yaxis()
            plt.legend(bbox_to_anchor=(0.97, .98),framealpha=0)
            plt.savefig(os.path.join(self.out_dir, 'estemated_all_ct_label.png'))
            plt.close()
            features['discrete_label_ct'].to_csv(os.path.join(self.out_dir, 'AllCellTypeLabel_nu' + str(nu) + '.csv'))

        ray.shutdown()
            


