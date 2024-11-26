import numpy as np
import pandas as pd

def ER(adata_st, cell_locations, region_type,region='region', single=True, ALL=False,region_mask=None):
    if ALL:
        cell_locations['count'] = 1
        er = []
        scale = adata_st.uns['spatial'][list(adata_st.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
        select_all = dict()
        for domain in region_type.keys():
                if domain != 'None':
                    select_all[domain] = np.array([],dtype=bool)
        for location in cell_locations[['x','y']].values:
            x, y = int(location[0] * scale), int(location[1] * scale)
            for domain in region_type.keys():
                if domain != 'None':
                    select_all[domain] = np.append(select_all[domain],(region_mask[domain][y,x] != 0))
        for domain in region_type.keys():
            if domain != 'None':
                select = select_all[domain]
                total_r = 0
                total = 0
                for label in region_type[domain]:
                    total_r += cell_locations.iloc[select].iloc[(cell_locations.iloc[select, :]['discrete_label_ct'] == label).values]['count'].sum()
                    total += cell_locations.iloc[(cell_locations['discrete_label_ct'] == label).values]['count'].sum()
                er.append(total_r / total)
            else:
                for label in region_type[domain]:
                    er.append(1 - (cell_locations['discrete_label_ct'] == label).sum() / cell_locations.shape[0])
    else:
        if single:
            spot_index = pd.DataFrame(cell_locations['spot_index'].value_counts())
            cell_locations['count'] = 1 / spot_index.loc[cell_locations['spot_index']].values
            #cell_locations['count'] = 1
            er = []
            for domain in region_type.keys():
                if domain != 'None':
                    select = adata_st.obs.index.values[np.where(adata_st.obs[region] == domain)[0]]
                    total_r = 0
                    total = 0
                    for label in region_type[domain]:
                        total_r += cell_locations.iloc[np.isin(cell_locations['spot_index'],select), :].iloc[(cell_locations.iloc[np.isin(cell_locations['spot_index'],select), :]['discrete_label_ct'] == label).values]['count'].sum()
                        total += cell_locations.iloc[(cell_locations['discrete_label_ct'] == label).values]['count'].sum()
                    er.append(total_r / total)
                else:
                    for label in region_type[domain]:
                        er.append(1 - (cell_locations['discrete_label_ct'] == label).sum() / cell_locations.shape[0])
        else:
            er = []
            for domain in region_type.keys():
                if domain != 'None':
                    select = adata_st.obs.index.values[np.where(adata_st.obs[region] == domain)[0]]
                    pred_ct = np.zeros(cell_locations.shape[0])
                    for label in region_type[domain]:
                        pred_ct += cell_locations.loc[:, label]
                    er.append(np.sum(pred_ct[select]) / np.sum(pred_ct))
                else:
                    for label in region_type[domain]:
                        er.append(1 - np.mean(cell_locations.loc[:, label]))
    return np.mean(er)
    
from sklearn.metrics import adjusted_rand_score

def ARI(adata_st, cell_locations, region_type, region='region',single=True, ALL=False,region_mask=None):
    scale = adata_st.uns['spatial'][list(adata_st.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    flipped_dict = {value: key for key, values in region_type.items() for value in values}
    if ALL:
        cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
        y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
        y_true = []
        for location in cell_locations[['x','y']].values:
            x, y = int(location[0] * scale), int(location[1] * scale)
            pp = False
            for domain in region_type.keys():
                if domain != 'None' and region_mask[domain][y,x] != 0:
                    y_true.append(domain)
                    pp = True
            if not pp:
                y_true.append('other')
    else:
        if single:
            cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
            y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
            y_true = adata_st.obs.loc[cell_locations['spot_index']][region].values.reshape(-1)
        else:
            spot_majorCT = cell_locations.columns.values[np.argmax(cell_locations,axis=1)]
            select = np.isin(spot_majorCT, list(flipped_dict.keys()))
            spot_majorCT = spot_majorCT[select]
            y_pred = [flipped_dict[dlc] for dlc in spot_majorCT]
            y_true = adata_st.obs.iloc[select][region].values.reshape(-1)
    return adjusted_rand_score(y_pred, y_true)

from collections import defaultdict

def Purtiy(adata_st, cell_locations, region_type, region='region',single=True,ALL=False,region_mask=None):
    scale = adata_st.uns['spatial'][list(adata_st.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    flipped_dict = {value: key for key, values in region_type.items() for value in values}
    if ALL:
        cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
        y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
        y_true = []
        for location in cell_locations[['x','y']].values:
            x, y = int(location[0] * scale), int(location[1] * scale)
            pp = False
            for domain in region_type.keys():
                if domain != 'None' and region_mask[domain][y,x] != 0:
                    y_true.append(domain)
                    pp = True
            if not pp:
                y_true.append('other')
    else:
        if single:
            cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
            y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
            y_true = adata_st.obs.loc[cell_locations['spot_index']][region].values.reshape(-1)
        else:
            spot_majorCT = cell_locations.columns.values[np.argmax(cell_locations,axis=1)]
            select = np.isin(spot_majorCT, list(flipped_dict.keys()))
            spot_majorCT = spot_majorCT[select]
            y_pred = [flipped_dict[dlc] for dlc in spot_majorCT]
            y_true = adata_st.obs.iloc[select][region].values.reshape(-1)

    contingency_matrix = defaultdict(lambda: defaultdict(int))

    for true, pred in zip(y_true, y_pred):
        contingency_matrix[pred][true] += 1
    purity = 0

    for cluster, cluster_dict in contingency_matrix.items():
        max_class_count = max(cluster_dict.values())
        purity += max_class_count

    purity /= len(y_pred)
    
    return purity

from sklearn.metrics import accuracy_score

def ACC(adata_st, cell_locations, region_type, region='region', single=True,ALL=False,region_mask=None):
    scale = adata_st.uns['spatial'][list(adata_st.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    flipped_dict = {value: key for key, values in region_type.items() for value in values}
    if ALL:
        cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
        y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
        y_true = []
        for location in cell_locations[['x','y']].values:
            x, y = int(location[0] * scale), int(location[1] * scale)
            pp = False
            for domain in region_type.keys():
                if domain != 'None' and region_mask[domain][y,x] != 0:
                    y_true.append(domain)
                    pp = True
            if not pp:
                y_true.append('other')
    else:
        if single:
            cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
            y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
            y_true = adata_st.obs.loc[cell_locations['spot_index']][region].values.reshape(-1)
        else:
            spot_majorCT = cell_locations.columns.values[np.argmax(cell_locations,axis=1)]
            select = np.isin(spot_majorCT, list(flipped_dict.keys()))
            spot_majorCT = spot_majorCT[select]
            y_pred = [flipped_dict[dlc] for dlc in spot_majorCT]
            y_true = adata_st.obs.iloc[select][region].values.reshape(-1)
    return accuracy_score(y_pred, y_true)

from sklearn.metrics import normalized_mutual_info_score
def NMI(adata_st, cell_locations, region_type, region='region', single=True,ALL=False,region_mask=None):
    flipped_dict = {value: key for key, values in region_type.items() for value in values}
    scale = adata_st.uns['spatial'][list(adata_st.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    flipped_dict = {value: key for key, values in region_type.items() for value in values}
    if ALL:
        cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
        y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
        y_true = []
        for location in cell_locations[['x','y']].values:
            x, y = int(location[0] * scale), int(location[1] * scale)
            pp = False
            for domain in region_type.keys():
                if domain != 'None' and region_mask[domain][y,x] != 0:
                    y_true.append(domain)
                    pp = True
            if not pp:
                y_true.append('other')
    else:
        if single:
            cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
            y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
            y_true = adata_st.obs.loc[cell_locations['spot_index']][region].values.reshape(-1)
        else:
            spot_majorCT = cell_locations.columns.values[np.argmax(cell_locations,axis=1)]
            select = np.isin(spot_majorCT, list(flipped_dict.keys()))
            spot_majorCT = spot_majorCT[select]
            y_pred = [flipped_dict[dlc] for dlc in spot_majorCT]
            y_true = adata_st.obs.iloc[select][region].values.reshape(-1)
    return normalized_mutual_info_score(y_true, y_pred)



from sklearn.metrics import f1_score
def F1(adata_st, cell_locations, region_type, region='region', single=True,ALL=False,region_mask=None):
    scale = adata_st.uns['spatial'][list(adata_st.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    flipped_dict = {value: key for key, values in region_type.items() for value in values}
    if ALL:
        cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
        y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
        y_true = []
        for location in cell_locations[['x','y']].values:
            x, y = int(location[0] * scale), int(location[1] * scale)
            pp = False
            for domain in region_type.keys():
                if domain != 'None' and region_mask[domain][y,x] != 0:
                    y_true.append(domain)
                    pp = True
            if not pp:
                y_true.append('other')
    else:
        if single:
            cell_locations = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], list(flipped_dict.keys()))]
            y_pred = [flipped_dict[dlc] for dlc in cell_locations['discrete_label_ct']]
            y_true = adata_st.obs.loc[cell_locations['spot_index']][region].values.reshape(-1)
        else:
            spot_majorCT = cell_locations.columns.values[np.argmax(cell_locations,axis=1)]
            select = np.isin(spot_majorCT, list(flipped_dict.keys()))
            spot_majorCT = spot_majorCT[select]
            y_pred = [flipped_dict[dlc] for dlc in spot_majorCT]
            y_true = adata_st.obs.iloc[select][region].values.reshape(-1)
    return f1_score(y_pred, y_true,average='macro')
