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
from shapely.validation import make_valid
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, lil_matrix
import cv2
from pathlib import Path

# Pre-compiled regex used for splitting digits out of strings.
re_digits = re.compile(r'(\d+)')


def toUpper(L):
    """
    Convert every string in a list to uppercase.

    Parameters
    ----------
    L : list[str]
        List of strings.

    Returns
    -------
    list[str]
        Uppercase-transformed strings.
    """
    return [s.upper() for s in L]


def embedded_numbers(s):
    """
    Split a string into alternating non-numeric / numeric parts.

    Parameters
    ----------
    s : str

    Returns
    -------
    list
        Mixed list of substrings and integers for natural sorting.
    """
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def sort_string(lst):
    """
    Sort strings using human-friendly (natural) order.
    """
    return sorted(lst, key=embedded_numbers)


def configure_logging(logger_name):
    """
    Configure a dual (file + stdout) logger named 'importer_logger'.

    Parameters
    ----------
    logger_name : str
        Prefix for the log filename.

    Returns
    -------
    logging.Logger
    """
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name + '.log'
    importer_logger = logging.getLogger('importer_logger')

    if not importer_logger.hasHandlers():
        importer_logger.setLevel(LOG_LEVEL)
        formatter = logging.Formatter('%(levelname)s : %(message)s')

        # File handler
        fh = logging.FileHandler(filename=log_filename)
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(formatter)
        importer_logger.addHandler(fh)

        # Stream handler (stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(LOG_LEVEL)
        sh.setFormatter(formatter)
        importer_logger.addHandler(sh)

    return importer_logger


def PlotVisiumGene(
    generated_cells,
    gene,
    size=0.8,
    alpha_img=0.3,
    perc=0.00,
    palette='rocket_r',
    vis_index=None,
    vis_index_only=None,
    colorbar_loc='right',
    title=None,
    ROI=None,
    keep_cell=None,
    limit=False,
    ax=None
):
    """
    Plot a single gene on Visium coordinates with percentile clipping.
    """
    test = generated_cells.copy()
    if keep_cell is not None:
        test = test[keep_cell, :]

    if 'spatial' not in test.obsm:
        if 'x' in test.obs.columns:
            test.obsm['spatial'] = test.obs[["x", "y"]].to_numpy()
        elif 'X' in test.obs.columns:
            test.obsm['spatial'] = test.obs[["X", "Y"]].to_numpy()
        spot_size = 30
    else:
        spot_size = None

    try:
        tmp = test[:, test.var.index == gene].X.toarray().copy()
    except Exception:
        tmp = test[:, test.var.index == gene].X.copy()

    # Percentile clipping + normalization to [0, 1].
    tmp = np.clip(tmp, np.quantile(tmp, perc), np.quantile(tmp, 1 - perc))
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())

    if vis_index is not None:
        tmp[~vis_index] = None
    if vis_index_only is not None:
        test = test[vis_index_only]
        tmp = tmp[vis_index_only]

    test.obs[gene + '_visual'] = tmp
    if title is None:
        title = '${}$'.format(gene)

    sc.pl.spatial(
        test,
        color=gene + '_visual',
        size=size,
        spot_size=spot_size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        na_color='#e3dede',
        cmap=palette,
        na_in_legend=False,
        colorbar_loc=colorbar_loc,
        ax=ax,
        title=title
    )

    sf = test.uns['spatial'][list(test.uns['spatial'].keys())[0]]['scalefactors']['tissue_hires_scalef']
    if limit and ROI is not None:
        ax.set_xlim([ROI['x_min'] * sf, ROI['x_max'] * sf])
        ax.set_ylim([ROI['y_max'] * sf, ROI['y_min'] * sf])

    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)


def PlotVisiumProp(
    adata,
    pivot,
    annotation_list,
    region_name=None,
    subregion=None,
    limit=False,
    ROI=None,
    lw=1,
    size=0.8,
    alpha_img=0.3,
    cmap='viridis',
    title='',
    ec='red',
    colorbar_loc=None,
    ax=None,
    **kwargs
):
    """
    Plot Visium proportions stored in a pivoted dataframe onto spatial coordinates.
    """
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
        ax=ax,
        title=title,
        sort_order=True,
        **kwargs
    )

    spatial = adata.obsm['spatial'].copy()
    if subregion is not None:
        select = adata.obs[region_name] == subregion
        spatial = spatial[select, :]
        print(spatial.shape)

    spatial_meta = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']
    sf = spatial_meta['tissue_hires_scalef']
    spot_diameter = spatial_meta.get('actual_spot_diameter_fullres', spatial_meta['spot_diameter_fullres'])
    spot_radius = spot_diameter / 2

    # Draw circle outlines for every spot.
    for sloc in spatial:
        circle = mpl.patches.Circle(
            (sloc[0] * sf, sloc[1] * sf),
            spot_radius * sf * size,
            ec=ec,
            lw=lw,
            fill=False
        )
        ax.add_patch(circle)

    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)

    if limit and ROI is not None:
        ax.set_xlim([ROI['x_min'] * sf, ROI['x_max'] * sf])
        ax.set_ylim([ROI['y_max'] * sf, ROI['y_min'] * sf])

    for _, spine in ax.spines.items():
        spine.set_visible(True)


def PlotVisiumRegion(
    adata,
    annotation_list,
    size=1,
    alpha_img=0.3,
    title_size=10,
    alpha=1,
    subset=None,
    palette='tab20',
    title='',
    legend=True,
    ax=None,
    **kwargs
):
    """
    Visualize categorical annotations on Visium data, optionally restricting to a subset.
    """
    test = adata.copy()

    if subset is not None:
        test.obs.loc[~test.obs[annotation_list].isin(subset), annotation_list] = None

    sc.pl.spatial(
        test,
        color=annotation_list,
        size=size,
        frameon=False,
        alpha_img=alpha_img,
        show=False,
        palette=palette,
        na_in_legend=False,
        ax=ax,
        alpha=alpha,
        title='',
        sort_order=True,
        **kwargs
    )
    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)

    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    if title:
        ax.set_title(title, fontsize=title_size)

    for _, spine in ax.spines.items():
        spine.set_visible(True)


def Ourpolygon(f, label, ax, ec='black'):
    """
    Overlay annotated polygons (e.g., ROIs) on an axis.
    """
    for shape in f['shapes']:
        if shape['label'] == label:
            polygon = patches.Polygon(
                shape['points'],
                closed=True,
                edgecolor=ec,
                facecolor='none',
                linewidth=2
            )
            ax.add_patch(polygon)


def PlotVisiumCells(
    adata_,
    annotation_list,
    size=0.8,
    alpha_img=0.3,
    alpha=0.5,
    lw=1,
    limit=False,
    ROI=None,
    title_size=10,
    subset=None,
    palette='tab20',
    ec='grey',
    ec_mask='pink',
    show_circle=True,
    title='',
    legend=True,
    ax=None,
    showlimit=True,
    fill=False,
    spot_list=None,
    keep_cell=None,
    Mask=False,
    Region=None,
    f=None,
    **kwargs
):
    """
    Plot per-cell annotations (stored in `adata.uns['cell_locations']`) on top of Visium spots.
    """
    adata = adata_.copy()

    if keep_cell is not None:
        adata.uns['cell_locations'] = adata.uns['cell_locations'].loc[
            np.intersect1d(keep_cell, adata.uns['cell_locations'].index)
        ]

    if ROI is not None and showlimit:
        spatial = adata.obsm["spatial"]
        mask = (
            (spatial[:, 0] > ROI['x_min']) &
            (spatial[:, 1] > ROI['y_min']) &
            (spatial[:, 0] < ROI['x_max']) &
            (spatial[:, 1] < ROI['y_max'])
        )
        adata = adata[mask, :].copy()

        cell_locs = adata.uns['cell_locations']
        adata.uns['cell_locations'] = cell_locs.loc[
            (cell_locs['x'] > ROI['x_min']) &
            (cell_locs['y'] > ROI['y_min']) &
            (cell_locs['x'] < ROI['x_max']) &
            (cell_locs['y'] < ROI['y_max'])
        ]

    merged_df = adata.uns['cell_locations'].copy()
    test = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    test.obsm['spatial'] = merged_df[["x", "y"]].to_numpy().copy()
    test.uns = adata.uns

    if subset is not None:
        test.obs.loc[~test.obs[annotation_list].isin(subset), annotation_list] = None

    sc.pl.spatial(
        test,
        color=annotation_list,
        size=size,
        alpha_img=alpha_img,
        show=False,
        palette=palette,
        na_in_legend=False,
        ax=ax,
        title='',
        sort_order=True,
        **kwargs
    )

    spatial_meta = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']
    sf = spatial_meta['tissue_hires_scalef']
    spot_diameter = spatial_meta.get('actual_spot_diameter_fullres', spatial_meta['spot_diameter_fullres'])
    spot_radius = spot_diameter / 2

    if show_circle:
        for sloc in adata.obsm['spatial']:
            circle = mpl.patches.Circle(
                (sloc[0] * sf, sloc[1] * sf),
                spot_radius * sf,
                ec=ec,
                lw=lw,
                fill=fill,
                fc=ec,
                alpha=alpha
            )
            ax.add_patch(circle)

        if spot_list is not None:
            adata_temp = adata.copy()
            adata_temp = adata_temp[np.intersect1d(spot_list, adata.obs_names), :]
            for sloc in adata_temp.obsm['spatial']:
                circle = mpl.patches.Circle(
                    (sloc[0] * sf, sloc[1] * sf),
                    spot_radius * sf,
                    ec=ec_mask,
                    lw=lw,
                    fill=fill,
                    fc=ec_mask,
                    alpha=alpha
                )
                ax.add_patch(circle)

    if Mask:
        Ourpolygon(f, Region, ax)

    if limit and ROI is not None:
        ax.set_xlim([ROI['x_min'] * sf, ROI['x_max'] * sf])
        ax.set_ylim([ROI['y_max'] * sf, ROI['y_min'] * sf])

    if not showlimit and ROI is not None:
        x_min = ROI['x_min'] * sf
        x_max = ROI['x_max'] * sf
        y_min = ROI['y_min'] * sf
        y_max = ROI['y_max'] * sf
        rect = mpl.patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=0.5,
            edgecolor='orange',
            linestyle='--',
            fill=False
        )
        ax.add_patch(rect)

    ax.axes.xaxis.label.set_visible(False)
    ax.axes.yaxis.label.set_visible(False)

    if title:
        ax.set_title(title, fontsize=title_size)
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    for _, spine in ax.spines.items():
        spine.set_visible(True)


def intersection_area(circle_center, circle_radius, polygon_points):
    """
    Compute the fraction of polygon area covered by the intersection with a circle.
    """
    circle = Point(circle_center).buffer(circle_radius)
    polygon = Polygon(polygon_points)
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    polygon_area = polygon.area
    intersection = circle.intersection(polygon)

    if intersection.is_empty:
        return 0.0
    if isinstance(intersection, Polygon):
        return intersection.area / polygon_area
    if intersection.geom_type == 'MultiPolygon':
        return sum(part.area for part in intersection.geoms) / polygon_area
    if intersection.geom_type == 'GeometryCollection':
        triangles_area = 0.0
        for geom in intersection.geoms:
            if geom.geom_type == 'Polygon':
                triangles_area += geom.area
            elif geom.geom_type == 'LineString':
                for i in range(len(geom.coords) - 1):
                    triangle = Polygon([geom.coords[i], geom.coords[i + 1], circle_center])
                    triangles_area += triangle.area
        return triangles_area / polygon_area
    return 0.0


def PlotRow_cell2cell(
    generated_cells_,
    CCC,
    L_gene,
    R_gene,
    L_ct,
    R_ct,
    color_dict,
    topk=100,
    ROI=None,
    title='',
    s=50,
    ax=None,
    invertY=True
):
    """
    Visualize top ligand-receptor interactions between two cell types with arrows.
    """
    if ROI is not None:
        mask = (
            (generated_cells_.obsm["spatial"][:, 0] > ROI['x_min']) &
            (generated_cells_.obsm["spatial"][:, 1] > ROI['y_min']) &
            (generated_cells_.obsm["spatial"][:, 0] < ROI['x_max']) &
            (generated_cells_.obsm["spatial"][:, 1] < ROI['y_max'])
        )
        generated_cells = generated_cells_[mask, :].copy()
        generated_cells.uns['cell_locations'] = generated_cells.uns['cell_locations'].loc[generated_cells.obs_names]
    else:
        generated_cells = generated_cells_.copy()

    all_L = generated_cells.uns['cell_locations'].loc[
        generated_cells.uns['cell_locations']['discrete_label_ct'] == L_ct, ['x', 'y']
    ]
    all_R = generated_cells.uns['cell_locations'].loc[
        generated_cells.uns['cell_locations']['discrete_label_ct'] == R_ct, ['x', 'y']
    ]
    other = generated_cells.uns['cell_locations'].loc[
        ~np.isin(generated_cells.uns['cell_locations']['discrete_label_ct'], [R_ct, L_ct]), ['x', 'y']
    ]

    LR = pd.DataFrame()
    for i in CCC.index:
        if CCC.loc[i, 'celltype_sender'] == L_ct and CCC.loc[i, 'celltype_receiver'] == R_ct:
            LR.loc[i, L_gene + '-' + R_gene] = CCC.loc[i, L_gene + '-' + R_gene]

    for idx in LR.index:
        cell1, cell2 = idx.split('/')
        LR.loc[idx, 'L_x'] = generated_cells_.uns['cell_locations'].loc[cell1, 'x']
        LR.loc[idx, 'L_y'] = generated_cells_.uns['cell_locations'].loc[cell1, 'y']
        LR.loc[idx, 'R_x'] = generated_cells_.uns['cell_locations'].loc[cell2, 'x']
        LR.loc[idx, 'R_y'] = generated_cells_.uns['cell_locations'].loc[cell2, 'y']

    topk_entries = LR.loc[LR.iloc[:, 0].nlargest(topk).index]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(11, 8), dpi=100, facecolor='#fafafa')

    ax.scatter(other['x'], other['y'], color='gray', label='Receptors', s=s * 0.5, alpha=0.1)
    ax.scatter(all_L['x'], all_L['y'], color=color_dict[L_ct], label='Receptors', s=s, alpha=0.5)
    ax.scatter(all_R['x'], all_R['y'], color=color_dict[R_ct], label='Ligands', s=s, alpha=0.5)

    for i in range(len(topk_entries)):
        L_x, L_y = topk_entries['L_x'][i], topk_entries['L_y'][i]
        R_x, R_y = topk_entries['R_x'][i], topk_entries['R_y'][i]
        if (
            ROI is None or
            (ROI['x_min'] < L_x < ROI['x_max'] and ROI['x_min'] < R_x < ROI['x_max'] and
             ROI['y_min'] < L_y < ROI['y_max'] and ROI['y_min'] < R_y < ROI['y_max'])
        ):
            arrow = FancyArrowPatch(
                (L_x, L_y),
                (R_x, R_y),
                connectionstyle="arc3,rad=0.3",
                arrowstyle='->',
                color='black',
                alpha=0.5,
                lw=3,
                mutation_scale=15
            )
            plt.gca().add_patch(arrow)

    plt.axis('off')
    plt.title(title)
    if invertY:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def constructNetwork(cell, K=10):
    """
    Build a symmetric KNN adjacency matrix (2D) for given cell coordinates.
    """
    coordinates = cell[['x', 'y']].values
    knn = NearestNeighbors(n_neighbors=K + 1)
    knn.fit(coordinates)
    _, indices = knn.kneighbors(coordinates)
    indices = indices[:, 1:]  # drop self

    num_points = coordinates.shape[0]
    row_indices = np.repeat(np.arange(num_points), K)
    col_indices = indices.flatten()
    data = np.ones_like(row_indices)

    adj_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_points, num_points))
    adj_matrix = (adj_matrix + adj_matrix.T).astype(bool).astype(int)
    return adj_matrix


def constructNetworkWithinSlice(cell, K=10):
    """
    Construct adjacency within a slice (3D coords), retaining edges only if labels match.
    """
    coordinates = cell[['new_z', 'new_x', 'new_y']].values
    labels = cell['label'].values
    knn = NearestNeighbors(n_neighbors=K + 1)
    knn.fit(coordinates)
    _, indices = knn.kneighbors(coordinates)
    indices = indices[:, 1:]

    label_mask = (labels[:, None] == labels[indices]).astype(int)

    row_indices = np.repeat(np.arange(coordinates.shape[0]), K)
    col_indices = indices.flatten()
    data = label_mask.flatten()

    adjacency_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(coordinates.shape[0], coordinates.shape[0]))
    return adjacency_matrix.toarray()


def constructNetworkBetweenSlices(cell1, cell2, K=10):
    """
    Build inter-slice adjacency (directed) preserving label consistency.
    """
    coordinates1 = cell1[['new_z', 'new_x', 'new_y']].values
    coordinates2 = cell2[['new_z', 'new_x', 'new_y']].values
    labels1 = cell1['label'].values
    labels2 = cell2['label'].values

    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(coordinates2)
    _, indices = knn.kneighbors(coordinates1)

    label_mask = (labels1[:, None] == labels2[indices]).astype(int)

    row_indices = np.repeat(np.arange(coordinates1.shape[0]), K)
    col_indices = indices.flatten()
    data = label_mask.flatten()

    adjacency_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(coordinates1.shape[0], coordinates2.shape[0])
    )
    return adjacency_matrix.toarray()


def constructFullNetwork(cell_table, slices, K=10):
    """
    Build a block adjacency matrix by combining intra-slice and inter-slice connections.
    """
    slice_adjacencies = []
    slice_sizes = []

    # Build intra-slice adjacency blocks.
    for slice_id in slices:
        cell_slice = cell_table[cell_table.index.str.startswith(slice_id)]
        adjacency_matrix = constructNetworkWithinSlice(cell_slice, K)
        slice_adjacencies.append(adjacency_matrix)
        slice_sizes.append(adjacency_matrix.shape[0])

    total_size = sum(slice_sizes)
    full_adjacency = lil_matrix((total_size, total_size))

    current_index = 0
    slice_indices = []
    for adjacency, size in zip(slice_adjacencies, slice_sizes):
        full_adjacency[current_index:current_index + size, current_index:current_index + size] = adjacency
        slice_indices.append(current_index)
        current_index += size

    # Connect adjacent slices with reduced K (K//2).
    for i in range(len(slices) - 1):
        cell_slice1 = cell_table[cell_table.index.str.startswith(slices[i])]
        cell_slice2 = cell_table[cell_table.index.str.startswith(slices[i + 1])]
        between_adjacency = constructNetworkBetweenSlices(cell_slice1, cell_slice2, K // 2)
        print(between_adjacency.sum())

        idx1_start = slice_indices[i]
        idx1_end = idx1_start + slice_sizes[i]
        idx2_start = slice_indices[i + 1]
        idx2_end = idx2_start + slice_sizes[i + 1]

        full_adjacency[idx1_start:idx1_end, idx2_start:idx2_end] = between_adjacency
        full_adjacency[idx2_start:idx2_end, idx1_start:idx1_end] = between_adjacency.T

    return csr_matrix(full_adjacency)


def estimate_transformation(A, B):
    """
    Estimate rigid transformation (rotation + translation) aligning A onto B.
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


def compute_transformed_bounds(image, R, t):
    """
    Compute axis-aligned bounds after applying a rigid transformation to an image.
    """
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]]).T
    transformed_corners = R @ corners + t[:, None]

    min_x, min_y = np.floor(transformed_corners.min(axis=1)).astype(int)
    max_x, max_y = np.ceil(transformed_corners.max(axis=1)).astype(int)
    return (min_x, min_y), (max_x, max_y)


def apply_transformation(image, R, t, global_min, global_max):
    """
    Apply a precomputed rigid transformation to an image and embed it into a global canvas.
    """
    h, w = image.shape[:2]
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([x.ravel(), y.ravel()], axis=1).T

    transformed_coords = (R @ coords + t[:, None]).T

    global_min_x, global_min_y = global_min
    global_max_x, global_max_y = global_max
    global_w = global_max_x - global_min_x
    global_h = global_max_y - global_min_y

    transformed_image = np.ones((int(global_h), int(global_w), 4), dtype=np.uint8) * 255

    for i in range(coords.shape[1]):
        orig_x, orig_y = coords[:, i]
        new_x, new_y = transformed_coords[i]

        global_x = int(new_x - global_min_x)
        global_y = int(new_y - global_min_y)

        if 0 <= global_x < global_w and 0 <= global_y < global_h:
            transformed_image[global_y, global_x, :] = image[orig_y, orig_x, :]

    return transformed_image


def process_images(image_paths, transformations):
    """
    Apply a list of rigid transformations to corresponding images and return both originals and transformed canvases.
    """
    global_min = np.array([np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf])

    # First pass: determine global canvas bounds.
    for image_path, (R, t) in zip(image_paths, transformations):
        image = cv2.imread(image_path[0], cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        if image.shape[2] == 3:
            alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            image = np.dstack([image, alpha_channel])
            image[(cv2.imread(image_path[1], cv2.IMREAD_UNCHANGED) == 0).all(axis=2), :3] = [255, 255, 255]

        min_coords, max_coords = compute_transformed_bounds(image, R, t)
        global_min = np.minimum(global_min, min_coords)
        global_max = np.maximum(global_max, max_coords)

    images = []
    transformed_images = []

    # Second pass: actually apply transformations.
    for image_path, (R, t) in zip(image_paths, transformations):
        print(f'Processing {image_path}')
        image = cv2.imread(image_path[0], cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        if image.shape[2] == 3:
            alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            image = np.dstack([image, alpha_channel])
            image[(cv2.imread(image_path[1], cv2.IMREAD_UNCHANGED) == 0).all(axis=2), :3] = [255, 255, 255]

        transformed_image = apply_transformation(image, R, t, global_min, global_max)
        images.append(image)
        transformed_images.append(transformed_image)

    return images, transformed_images