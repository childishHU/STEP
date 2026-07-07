import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    normalized_mutual_info_score,
    f1_score,
)

def _resolve_true_pred(adata_st, cell_locations, region_type,
                       region='region', single=True, ALL=False, region_mask=None):
    """
    Internal utility that reproduces the original label-alignment logic.
    Returns the ground-truth domains (y_true) and predicted domains (y_pred).
    """
    spatial_key = list(adata_st.uns['spatial'].keys())[0]
    scale = adata_st.uns['spatial'][spatial_key]['scalefactors']['tissue_hires_scalef']

    flipped_dict = {value: key for key, values in region_type.items() for value in values}
    valid_labels = list(flipped_dict.keys())

    if ALL:
        filtered = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], valid_labels)]
        y_pred = [flipped_dict[label] for label in filtered['discrete_label_ct']]
        y_true = []

        # Map each cell to a domain via the high-resolution masks
        for x_raw, y_raw in filtered[['x', 'y']].values:
            x = int(x_raw * scale)
            y = int(y_raw * scale)
            matched = False
            for domain in region_type.keys():
                if domain != 'None' and region_mask[domain][y, x] != 0:
                    y_true.append(domain)
                    matched = True
            if not matched:
                y_true.append('other')
        return y_true, y_pred

    if single:
        filtered = cell_locations.iloc[np.isin(cell_locations['discrete_label_ct'], valid_labels)]
        y_pred = [flipped_dict[label] for label in filtered['discrete_label_ct']]
        y_true = adata_st.obs.loc[filtered['spot_index']][region].values.reshape(-1)
        return y_true, y_pred

    # Probabilistic mode: use argmax over probability columns
    spot_major = cell_locations.columns.values[np.argmax(cell_locations, axis=1)]
    select = np.isin(spot_major, valid_labels)
    spot_major = spot_major[select]
    y_pred = [flipped_dict[label] for label in spot_major]
    y_true = adata_st.obs.iloc[select][region].values.reshape(-1)
    return y_true, y_pred


def ER(adata_st, cell_locations, region_type, region='region', single=True, ALL=False, region_mask=None):
    """
    Compute enrichment ratios (ER) for groups of cell-type labels within spatial domains.
    """
    def append_none_domain(er_list, labels):
        """Append complement proportions for the 'None' background domain."""
        total_cells = cell_locations.shape[0]
        for label in labels:
            er_list.append(1 - (cell_locations['discrete_label_ct'] == label).sum() / total_cells)

    if ALL:
        cell_locations['count'] = 1
        er = []

        spatial_key = list(adata_st.uns['spatial'].keys())[0]
        scale = adata_st.uns['spatial'][spatial_key]['scalefactors']['tissue_hires_scalef']

        select_all = {
            domain: np.array([], dtype=bool)
            for domain in region_type.keys()
            if domain != 'None'
        }

        for x_raw, y_raw in cell_locations[['x', 'y']].values:
            x = int(x_raw * scale)
            y = int(y_raw * scale)
            for domain in select_all.keys():
                select_all[domain] = np.append(select_all[domain], region_mask[domain][y, x] != 0)

        for domain, labels in region_type.items():
            if domain == 'None':
                append_none_domain(er, labels)
                continue

            select = select_all[domain]
            total_r = 0
            total = 0
            for label in labels:
                subset = cell_locations.iloc[select]
                mask = (subset['discrete_label_ct'] == label).values
                total_r += subset.iloc[mask]['count'].sum()

                mask_all = (cell_locations['discrete_label_ct'] == label).values
                total += cell_locations.iloc[mask_all]['count'].sum()

            er.append(total_r / total)

    else:
        if single:
            er = []
            spot_index = pd.DataFrame(cell_locations['spot_index'].value_counts())
            cell_locations['count'] = 1 / spot_index.loc[cell_locations['spot_index']].values

            for domain, labels in region_type.items():
                if domain == 'None':
                    append_none_domain(er, labels)
                    continue

                select = adata_st.obs.index.values[np.where(adata_st.obs[region] == domain)[0]]
                domain_mask = np.isin(cell_locations['spot_index'], select)

                total_r = 0
                total = 0
                for label in labels:
                    subset = cell_locations.iloc[domain_mask, :]
                    mask = (subset['discrete_label_ct'] == label).values
                    total_r += subset.iloc[mask]['count'].sum()

                    mask_all = (cell_locations['discrete_label_ct'] == label).values
                    total += cell_locations.iloc[mask_all]['count'].sum()

                er.append(total_r / total)
        else:
            er = []
            for domain, labels in region_type.items():
                if domain == 'None':
                    append_none_domain(er, labels)
                    continue

                select = adata_st.obs.index.values[np.where(adata_st.obs[region] == domain)[0]]
                pred_ct = np.zeros(cell_locations.shape[0])
                for label in labels:
                    pred_ct += cell_locations.loc[:, label]

                er.append(np.sum(pred_ct[select]) / np.sum(pred_ct))

    return np.mean(er)


def ARI(adata_st, cell_locations, region_type, region='region', single=True, ALL=False, region_mask=None):
    """
    Adjusted Rand Index between predicted domains and true domains.
    """
    y_true, y_pred = _resolve_true_pred(
        adata_st, cell_locations, region_type, region, single, ALL, region_mask
    )
    return adjusted_rand_score(y_pred, y_true)


def Purtiy(adata_st, cell_locations, region_type, region='region', single=True, ALL=False, region_mask=None):
    """
    Clustering purity between predicted domains and true domains.
    """
    y_true, y_pred = _resolve_true_pred(
        adata_st, cell_locations, region_type, region, single, ALL, region_mask
    )

    contingency = defaultdict(lambda: defaultdict(int))
    for true_label, pred_label in zip(y_true, y_pred):
        contingency[pred_label][true_label] += 1

    purity = 0
    for counts in contingency.values():
        purity += max(counts.values())

    purity /= len(y_pred)
    return purity


def ACC(adata_st, cell_locations, region_type, region='region', single=True, ALL=False, region_mask=None):
    """
    Classification accuracy between predicted domains and true domains.
    """
    y_true, y_pred = _resolve_true_pred(
        adata_st, cell_locations, region_type, region, single, ALL, region_mask
    )
    return accuracy_score(y_pred, y_true)


def NMI(adata_st, cell_locations, region_type, region='region', single=True, ALL=False, region_mask=None):
    """
    Normalized Mutual Information between predicted domains and true domains.
    """
    y_true, y_pred = _resolve_true_pred(
        adata_st, cell_locations, region_type, region, single, ALL, region_mask
    )
    return normalized_mutual_info_score(y_true, y_pred)


def F1(adata_st, cell_locations, region_type, region='region', single=True, ALL=False, region_mask=None):
    """
    Macro-averaged F1 score between predicted domains and true domains.
    """
    y_true, y_pred = _resolve_true_pred(
        adata_st, cell_locations, region_type, region, single, ALL, region_mask
    )
    return f1_score(y_pred, y_true, average='macro')