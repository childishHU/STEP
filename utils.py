import scanpy as sc
# import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# import skimage
import logging
import re
from random import sample
import sys
from typing import NoReturn
from shapely.geometry import Point, Polygon
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

re_digits = re.compile(r'(\d+)')


# Capitalize the string in the string array
def toUpper(L):
    """
    :param L: List

    :return: List
    """
    L_upper = []
    for s in L:
        L_upper.append(s.upper())
    return L_upper

def embedded_numbers(s):
    """
    :param s: String, a string to be decomposed.
    :return: A list
    """
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def sort_string(lst):
    return sorted(lst, key=embedded_numbers)

def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger

def PlotVisiumGene(generated_cells,gene,size=0.8,alpha_img=0.3,perc=0.00,palette='rocket_r', vis_index = None, vis_index_only = None, colorbar_loc='right',title=None,ROI=None, keep_cell=None,limit=False, ax=None):
    
    test = generated_cells.copy()
    if keep_cell is not None:
        test = test[keep_cell,:]
    if not 'spatial' in test.obsm.keys():
        if 'x' in test.obs.columns:
            test.obsm['spatial'] = test.obs[["x", "y"]].to_numpy()
        elif 'X' in test.obs.columns:
            test.obsm['spatial'] = test.obs[["X", "Y"]].to_numpy()

        spot_size = 30
    else:
        spot_size = None
        
    try:
        tmp = test[:,test.var.index==gene].X.toarray().copy()
    except:
        tmp = test[:,test.var.index==gene].X.copy()
        
    tmp = np.clip(tmp,np.quantile(tmp,perc),np.quantile(tmp,1-perc))
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    if vis_index is not None:
        tmp[~vis_index] = None
    if vis_index_only is not None:
        test = test[vis_index_only]
        tmp = tmp[vis_index_only]
    test.obs[gene+'_visual'] = tmp
    if title is None:
        title='${}$'.format(gene)
    
    sc.pl.spatial(
        test,
        color=gene+'_visual',
        size=size,
        spot_size=spot_size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        na_color='#e3dede',
        cmap=palette,
        na_in_legend=False,
        colorbar_loc=colorbar_loc,
        ax=ax,title=title
    )
    sf = test.uns['spatial'][list(test.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    if limit and ROI is not None:
        ax.set_xlim([ROI['x_min'] * sf, ROI['x_max'] * sf])  
        ax.set_ylim([ROI['y_max'] * sf, ROI['y_min'] * sf])  
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)


def PlotVisiumProp(adata,pivot,annotation_list,region_name=None,subregion=None,limit=False,ROI=None,lw=1,size=0.8,alpha_img=0.3,cmap='viridis', title='',ec='red', colorbar_loc=None,ax=None,**kwargs):

    test = adata.copy()
    test.obs = pivot.loc[test.obs.index]
    
        
    sc.pl.spatial(
        test,
        color=annotation_list,
        size=size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        cmap=cmap,
        colorbar_loc=colorbar_loc,
        na_in_legend=False,
        ax=ax,title=title,sort_order=True,**kwargs
    )
    spatial = adata.obsm['spatial'].copy()
    if subregion is not None:
        select = adata.obs[region_name] == subregion
        spatial = spatial[select,:]
        print(spatial.shape)
    sf = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    spot_radius = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']/2
    for sloc in spatial:
        rect = mpl.patches.Circle(
            (sloc[0] * sf, sloc[1] * sf),
            spot_radius * sf * size,
            ec=ec,
            lw=lw,
            fill=False
        )
        ax.add_patch(rect)
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
    if limit and ROI is not None:
        ax.set_xlim([ROI['x_min'] * sf, ROI['x_max'] * sf])  
        ax.set_ylim([ROI['y_max'] * sf, ROI['y_min'] * sf])    

    
    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)

def PlotVisiumRegion(adata,annotation_list,size=1,alpha_img=0.3,title_size=10,alpha=1,subset=None,palette='tab20', title='',legend = True, ax=None,**kwargs):

    test = adata.copy()

    if subset is not None:
        #test = test[test.obs[annotation_list].isin(subset)]
        test.obs.loc[~test.obs[annotation_list].isin(subset),annotation_list] = None
        
    sc.pl.spatial(
        test,
        color=annotation_list,
        size=size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        palette=palette,
        na_in_legend=False,
        ax=ax,alpha=alpha,title='',sort_order=True,**kwargs
    )
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
    
    if not legend:
        ax.get_legend().remove()
    if title != '':
        ax.set_title(title, fontsize=title_size)
    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)

def PlotVisiumCells(adata_,annotation_list,size=0.8,alpha_img=0.3,alpha=0.5,lw=1,limit=False,ROI=None,title_size=10,subset=None,palette='tab20',
                    ec='grey',ec_mask='pink',show_circle = True, title='',legend = True, ax=None,showlimit=True,
                    fill=False,spot_list=None,keep_cell=None,**kwargs):
    adata = adata_.copy()
    if keep_cell is not None:
        adata.uns['cell_locations'] =  adata.uns['cell_locations'].loc[keep_cell]
    if ROI is not None and showlimit:
        adata = adata[
            (adata.obsm["spatial"][:, 0] > ROI['x_min']) & (adata.obsm["spatial"][:, 1] >  ROI['y_min'])
            & (adata.obsm["spatial"][:, 0] <  ROI['x_max']) & (adata.obsm["spatial"][:, 1] <  ROI['y_max']), :
        ].copy()
        adata.uns['cell_locations'] = adata.uns['cell_locations'].loc[
            (adata.uns['cell_locations']['x'] > ROI['x_min']) & (adata.uns['cell_locations']['y'] >  ROI['y_min'])
            & (adata.uns['cell_locations']['x'] <  ROI['x_max']) & (adata.uns['cell_locations']['y'] <  ROI['y_max'])
        , :]

    merged_df = adata.uns['cell_locations'].copy()
    test = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    test.obsm['spatial'] = merged_df[["x", "y"]].to_numpy().copy()
    test.uns = adata.uns
    
    if subset is not None:
        #test = test[test.obs[annotation_list].isin(subset)]
        test.obs.loc[~test.obs[annotation_list].isin(subset),annotation_list] = None
        
    sc.pl.spatial(
        test,
        color=annotation_list,
        size=size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        palette=palette,
        na_in_legend=False,
        ax=ax,title='',sort_order=True,**kwargs
    )
    sf = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    if show_circle:
        spot_radius = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']['spot_diameter_fullres']/2
        
        for sloc in adata.obsm['spatial']:
            rect = mpl.patches.Circle(
                (sloc[0] * sf, sloc[1] * sf),
                spot_radius * sf,
                ec=ec,
                lw=lw,
                fill=fill,
                fc=ec,
                alpha=alpha
            )
            ax.add_patch(rect)
        if spot_list is not None:
            adata_temp = adata.copy()
            adata_temp = adata_temp[np.intersect1d(spot_list, adata.obs_names),:]
            for sloc in adata_temp.obsm['spatial']:
                rect = mpl.patches.Circle(
                    (sloc[0] * sf, sloc[1] * sf),
                    spot_radius * sf,
                    ec=ec_mask,
                    lw=lw,
                    fill=fill,
                    fc=ec_mask,
                    alpha=alpha
                )
                ax.add_patch(rect)
            
    if limit and ROI is not None:
        ax.set_xlim([ROI['x_min'] * sf, ROI['x_max'] * sf])  
        ax.set_ylim([ROI['y_max'] * sf, ROI['y_min'] * sf])  
    if not showlimit and ROI is not None:
        x_min = ROI['x_min'] * sf
        x_max = ROI['x_max'] * sf
        y_min = ROI['y_min'] * sf
        y_max = ROI['y_max'] * sf
        rect = mpl.patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=0.5, edgecolor='orange', linestyle='--', fill=False)
        ax.add_patch(rect)
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)
    if title != '':
        ax.set_title(title, fontsize=title_size)
    if not legend:
        ax.get_legend().remove()
    
    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)




def intersection_area(circle_center, circle_radius, polygon_points):

    circle = Point(circle_center).buffer(circle_radius)
    polygon = Polygon(polygon_points)
    polygon_area = polygon.area
    intersection = circle.intersection(polygon)

    if intersection.is_empty:
        return 0.0
    if isinstance(intersection, Polygon):
        return intersection.area / polygon_area

    if intersection.geom_type == 'MultiPolygon':
        return sum(part.area for part in intersection.geoms) / polygon_area
    elif intersection.geom_type == 'GeometryCollection':
        triangles_area = 0.0
        for geom in intersection.geoms:
            if geom.geom_type == 'Polygon':
                triangles_area += geom.area
            elif geom.geom_type == 'LineString':
                for i in range(len(geom.coords) - 1):
                    triangle = Polygon([geom.coords[i], geom.coords[i+1], circle_center])
                    triangles_area += triangle.area
        return triangles_area / polygon_area
    else:
        return 0.0 

from matplotlib.patches import FancyArrowPatch
def PlotRow_cell2cell(generated_cells_,CCC,L_gene,R_gene,L_ct,R_ct,color_dict,topk=100,ROI=None,title='',s=50,ax=None,invertY=True):
    if ROI is not None:
        generated_cells = generated_cells_[
            (generated_cells_.obsm["spatial"][:, 0] > ROI['x_min']) & (generated_cells_.obsm["spatial"][:, 1] >  ROI['y_min'])
            & (generated_cells_.obsm["spatial"][:, 0] <  ROI['x_max']) & (generated_cells_.obsm["spatial"][:, 1] <  ROI['y_max']), :
        ].copy()
        generated_cells.uns['cell_locations'] = generated_cells.uns['cell_locations'].loc[generated_cells.obs_names]
    else:
        generated_cells = generated_cells_.copy()
    all_L = pd.DataFrame(generated_cells.uns['cell_locations'].loc[generated_cells.uns['cell_locations']['discrete_label_ct'] == L_ct,['x','y']])
    all_R = pd.DataFrame(generated_cells.uns['cell_locations'].loc[generated_cells.uns['cell_locations']['discrete_label_ct'] == R_ct,['x','y']])
    other = pd.DataFrame(generated_cells.uns['cell_locations'].loc[~np.isin(generated_cells.uns['cell_locations']['discrete_label_ct'],[R_ct,L_ct]),['x','y']])
    LR = pd.DataFrame()

    for i in CCC.index:
        if CCC.loc[i, 'celltype_sender'] == L_ct and CCC.loc[i, 'celltype_receiver'] == R_ct:
            LR.loc[i, L_gene + '-' + R_gene] = CCC.loc[i, L_gene + '-' + R_gene]
    for idx in LR.index:
        cell1, cell2 = idx.split('/')
        LR.loc[idx,'L_x'] = generated_cells_.uns['cell_locations'].loc[cell1,'x']
        LR.loc[idx,'L_y'] = generated_cells_.uns['cell_locations'].loc[cell1,'y']
        LR.loc[idx,'R_x'] = generated_cells_.uns['cell_locations'].loc[cell2,'x']
        LR.loc[idx,'R_y'] = generated_cells_.uns['cell_locations'].loc[cell2,'y']
    topk = LR.loc[LR.iloc[:,0].nlargest(topk).index]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 8),dpi=100,facecolor='#fafafa')
    ax.scatter(other['x'], other['y'], color='gray', label='Receptors', s=s*0.5, alpha=0.1)
    ax.scatter(all_L['x'], all_L['y'], color=color_dict[L_ct], label='Receptors', s=s, alpha=0.5)
    ax.scatter(all_R['x'], all_R['y'], color=color_dict[R_ct], label='Ligands', s=s, alpha=0.5)
    
    for i in range(len(topk)):
        
        L_x, L_y = topk['L_x'][i], topk['L_y'][i]
        R_x, R_y = topk['R_x'][i], topk['R_y'][i]
        mid_x = (L_x + R_x) / 2
        mid_y = (L_y + R_y) / 2
        if (ROI is None) or (L_x > ROI['x_min'] and L_x < ROI['x_max'] and R_x > ROI['x_min'] and R_x < ROI['x_max'] and L_y > ROI['y_min'] and L_y < ROI['y_max'] and R_y > ROI['y_min'] and R_y < ROI['y_max']):
            arrow = FancyArrowPatch((L_x, L_y), (R_x, R_y), 
                                    connectionstyle="arc3,rad=0.3", 
                                    arrowstyle='->', color='black', alpha=0.5,lw=3,mutation_scale=15)
            
            plt.gca().add_patch(arrow)
    plt.axis('off')
    plt.title(title)
    if invertY:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

