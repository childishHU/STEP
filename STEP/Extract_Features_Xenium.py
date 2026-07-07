# -*- coding: utf-8 -*-
import os
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
from skimage import io as skio
from skimage.draw import polygon as sk_polygon
from shapely.geometry import Polygon, MultiPolygon
from scipy.ndimage import distance_transform_edt

import pyarrow as pa
import pyarrow.parquet as pq

# ------------------ Paths ------------------
IMAGE_PATH = '/ZJU/data1/HZQ/Human_Colon_Cancer/data/Xenium/p2/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_he_image.ome.tif'
CELL_PARQUET = '/ZJU/data1/HZQ/Human_Colon_Cancer/data/Xenium/p2/cell_boundaries.parquet'
NUC_PARQUET  = '/ZJU/data1/HZQ/Human_Colon_Cancer/data/Xenium/p2/nucleus_boundaries.parquet'
PROB_PATH = None
SAVE_DIR = '/ZJU/data1/HZQ/revision2/benchmarking/xenium_crc_h5ad'
PARQUET_OUT = os.path.join(SAVE_DIR, "features.parquet")

# ------------------ Parameters ------------------
SCALE = 0.2125
MEMBRANE_WIDTH = 3
ENABLE_CELL_EXPAND_IF_MISSING = True
CELL_EXPANSION_FACTOR = 3.0
CELL_CONSTRAIN_SCALE = 1.0
H_ALIGN = pd.read_csv('/ZJU/data1/HZQ/Human_Colon_Cancer/data/Visium_CytAssist/p2/Visium_V2_Human_Colon_Cancer_P2_tissue_image_alignment_files/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_he_imagealignment.csv', header=None).values

# Threading and batching
NUM_WORKERS = 4
BATCH_SIZE = 4096

# Local H&E percentile stretching
LOCAL_P_LOW = 1
LOCAL_P_HIGH = 99
LOCAL_PERCENTILE_SAMPLE_STEP = 4  # 4~8 is a good trade-off between speed and robustness

# Add small padding to bbox to stabilize morphological ops near edges
BBOX_PAD = 8

# ------------------ Utils ------------------
def transform_points_inverse(points_xy_t, M):
    M_inv = np.linalg.inv(M)
    ones = np.ones((points_xy_t.shape[0], 1), dtype=points_xy_t.dtype)
    pts_h_t = np.hstack([points_xy_t, ones])
    pts_h = pts_h_t @ M_inv.T
    pts_xy = pts_h[:, :2] / pts_h[:, 2:3]
    return pts_xy

def apply_homography(pts, H):
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    warp = (H @ pts_h.T).T
    warp = warp[:, :2] / warp[:, 2:3]
    return warp

def rgb_to_he_qupath_local(rgb_patch, p_low=1, p_high=99, sample_step=4):
    """
    Local H&E color deconvolution with percentile stretching on a patch.
    Returns uint8 H, E images for this patch only (avoids whole-slide computation).
    """
    if rgb_patch.ndim == 2:
        rgb = np.repeat(rgb_patch[..., None], 3, axis=2).astype(np.float32, copy=False)
    else:
        rgb = rgb_patch[..., :3]
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32, copy=False)

    # Convert to optical density (OD)
    np.clip(rgb, 1.0, 255.0, out=rgb)
    rgb *= (1.0 / 255.0)
    OD = -np.log(rgb + 1e-8, dtype=np.float32)

    # QuPath stain vectors
    Hv = np.array([0.60968, 0.65246, 0.45010], dtype=np.float32)
    Ev = np.array([0.21306, 0.87722, 0.43022], dtype=np.float32)
    Hv /= (np.linalg.norm(Hv) + 1e-8)
    Ev /= (np.linalg.norm(Ev) + 1e-8)
    Cv = np.cross(Hv, Ev).astype(np.float32)
    Cv /= (np.linalg.norm(Cv) + 1e-8)
    M = np.stack([Hv, Ev, Cv], axis=1)  # 3x3

    od_stains = np.einsum('ijk,kl->ijl', OD, M, dtype=np.float32, optimize=True)
    H_img = od_stains[..., 0]
    E_img = od_stains[..., 1]

    # Percentile stretching within the patch
    if sample_step > 1:
        Hs = H_img[::sample_step, ::sample_step].ravel()
        Es = E_img[::sample_step, ::sample_step].ravel()
    else:
        Hs = H_img.ravel()
        Es = E_img.ravel()

    H_lo, H_hi = np.percentile(Hs, [p_low, p_high]) if Hs.size else (0.0, 1.0)
    E_lo, E_hi = np.percentile(Es, [p_low, p_high]) if Es.size else (0.0, 1.0)
    eps = 1e-6
    H8 = ((np.clip(H_img, H_lo, H_hi) - H_lo) * (255.0 / max(H_hi - H_lo, eps))).astype(np.uint8)
    E8 = ((np.clip(E_img, E_lo, E_hi) - E_lo) * (255.0 / max(E_hi - E_lo, eps))).astype(np.uint8)
    return H8, E8

def circularity(area, perimeter):
    if perimeter is None or perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter ** 2))

def safe_stats(vals):
    if vals.size == 0:
        return dict(mean=0.0, median=0.0, min=0.0, max=0.0, std=0.0)
    return dict(
        mean=float(vals.mean()),
        median=float(np.median(vals)),
        min=float(vals.min()),
        max=float(vals.max()),
        std=float(vals.std()),
    )

def rasterize_polygon_local(poly, H, W, bbox=None):
    """
    Rasterize a shapely polygon to a binary mask.
    If bbox=(xmin,ymin,xmax,ymax) is provided, draw in the local bbox coordinates and
    return the local mask and its top-left offset in the global image.
    """
    if bbox is None:
        mask = np.zeros((H, W), dtype=np.uint8)
        if poly is None:
            return mask, (0, 0)
        def draw_one_polygon(p: Polygon):
            exterior = np.array(p.exterior.coords)
            rr, cc = sk_polygon(exterior[:,1], exterior[:,0], shape=mask.shape)
            mask[rr, cc] = 1
            for hole in p.interiors:
                inner = np.array(hole.coords)
                rr_h, cc_h = sk_polygon(inner[:,1], inner[:,0], shape=mask.shape)
                mask[rr_h, cc_h] = 0
        if isinstance(poly, Polygon):
            draw_one_polygon(poly)
        elif isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                draw_one_polygon(p)
        return mask, (0, 0)

    xmin, ymin, xmax, ymax = bbox
    w = max(int(xmax - xmin), 1)
    h = max(int(ymax - ymin), 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly is None:
        return mask, (int(xmin), int(ymin))
    def draw_one_polygon_local(p: Polygon):
        exterior = np.array(p.exterior.coords)
        ex = exterior[:, 0] - xmin
        ey = exterior[:, 1] - ymin
        rr, cc = sk_polygon(ey, ex, shape=mask.shape)
        mask[rr, cc] = 1
        for hole in p.interiors:
            inner = np.array(hole.coords)
            ix = inner[:, 0] - xmin
            iy = inner[:, 1] - ymin
            rr_h, cc_h = sk_polygon(iy, ix, shape=mask.shape)
            mask[rr_h, cc_h] = 0
    if isinstance(poly, Polygon):
        draw_one_polygon_local(poly)
    elif isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            draw_one_polygon_local(p)
    return mask, (int(xmin), int(ymin))

def polygon_from_group(df_grp):
    """
    Assemble a polygon from a group's vertices. Assumes a single outer ring.
    Closes the ring if last point != first point. Fixes invalid geometry via buffer(0).
    """
    pts = df_grp[['vertex_x', 'vertex_y']].to_numpy()
    if len(pts) < 3:
        return None
    if not (np.isclose(pts[0], pts[-1]).all()):
        pts = np.vstack([pts, pts[0]])
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None

def compute_morphology(mask):
    """
    Compute morphology metrics from a binary mask:
    area, perimeter, solidity, major/minor axis (ellipse/minAreaRect), length (=major),
    circularity, max/min diameter (same as major/minor).
    """
    binmask = (mask > 0).astype(np.uint8) * 255
    area = float((mask > 0).sum())
    perimeter = 0.0
    solidity = 0.0
    major = 0.0
    minor = 0.0

    cnts, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(c, True))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        if len(c) >= 5:
            (x0, y0), (MA, ma), angle = cv2.fitEllipse(c)
            major = float(max(MA, ma))
            minor = float(min(MA, ma))
        else:
            rect = cv2.minAreaRect(c)
            (w, h) = rect[1]
            major = float(max(w, h))
            minor = float(min(w, h))
    length = major
    circ = circularity(area, perimeter)
    max_diam = major
    min_diam = minor
    return area, perimeter, solidity, major, minor, length, circ, max_diam, min_diam

def expand_nucleus_to_cell(nuc_mask, expansion=CELL_EXPANSION_FACTOR, constrain_scale=CELL_CONSTRAIN_SCALE):
    """
    Synthesize a cell mask by expanding the nucleus mask (QuPath-like behavior).
    Effective expansion is limited by constrain_scale * nucleus radius.
    """
    if nuc_mask.max() == 0:
        return nuc_mask.copy()
    area = float((nuc_mask > 0).sum())
    r = np.sqrt(max(area, 1.0) / np.pi)
    eff_expansion = min(expansion, constrain_scale * r)
    inv = (nuc_mask == 0).astype(np.uint8)
    dist = distance_transform_edt(inv)
    cell_mask = ((nuc_mask > 0) | (dist <= eff_expansion)).astype(np.uint8)
    return cell_mask

def polygon_radius_from_poly(geom):
    """
    Circumcircle radius proxy: max Euclidean distance from centroid
    to exterior vertices of the primary polygon.
    - Accepts Polygon or MultiPolygon (uses the largest polygon by area).
    - Returns 0.0 if geometry is None/empty or has insufficient vertices.
    """
    if geom is None:
        return 0.0
    if isinstance(geom, Polygon):
        poly = None if geom.is_empty else geom
    elif isinstance(geom, MultiPolygon):
        if geom.is_empty or len(geom.geoms) == 0:
            poly = None
        else:
            poly = max(geom.geoms, key=lambda g: g.area)
    else:
        try:
            g2 = geom.buffer(0)
            return polygon_radius_from_poly(g2)
        except Exception:
            return 0.0

    if poly is None or poly.is_empty:
        return 0.0

    coords = np.array(poly.exterior.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return 0.0
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    c = np.array(poly.centroid.coords[0], dtype=np.float64)
    d = np.linalg.norm(coords - c, axis=1)
    return float(d.max()) if d.size else 0.0

def polygon_string_from_poly(geom):
    """
    Serialize exterior vertices of a Polygon or MultiPolygon to
    a string like '[(x1,y1),(x2,y2),...]'.
    - If MultiPolygon: use the largest polygon by area.
    - Return empty string if geometry is None or empty.
    """
    if geom is None:
        return ""
    if isinstance(geom, Polygon):
        poly = geom if not geom.is_empty else None
    elif isinstance(geom, MultiPolygon):
        if geom.is_empty or len(geom.geoms) == 0:
            poly = None
        else:
            poly = max(geom.geoms, key=lambda g: g.area)
    else:
        try:
            g2 = geom.buffer(0)
            return polygon_string_from_poly(g2)
        except Exception:
            return ""

    if poly is None or poly.is_empty:
        return ""

    coords = np.array(poly.exterior.coords, dtype=np.float64)
    if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    pairs = ",".join(f"({x},{y})" for x, y in coords)
    return f"[{pairs}]"

FEATURE_NAMES = [
    "Detection probability","x","y",
    "Nucleus: Area (px)","Nucleus: Length (px)","Nucleus: Circularity","Nucleus: Solidity","Nucleus: Max diameter (px)","Nucleus: Min diameter (px)","Nucleus/Cell area ratio",
    "Cell: Area (px)","Cell: Length (px)","Cell: Circularity","Cell: Solidity","Cell: Max diameter (px)","Cell: Min diameter (px)",
    "Hematoxylin: Nucleus: Mean","Hematoxylin: Nucleus: Median","Hematoxylin: Nucleus: Min","Hematoxylin: Nucleus: Max","Hematoxylin: Nucleus: Std.Dev.",
    "Hematoxylin: Cytoplasm: Mean","Hematoxylin: Cytoplasm: Median","Hematoxylin: Cytoplasm: Min","Hematoxylin: Cytoplasm: Max","Hematoxylin: Cytoplasm: Std.Dev.",
    "Hematoxylin: Membrane: Mean","Hematoxylin: Membrane: Median","Hematoxylin: Membrane: Min","Hematoxylin: Membrane: Max","Hematoxylin: Membrane: Std.Dev.",
    "Hematoxylin: Cell: Mean","Hematoxylin: Cell: Median","Hematoxylin: Cell: Min","Hematoxylin: Cell: Max","Hematoxylin: Cell: Std.Dev.",
    "Eosin: Nucleus: Mean","Eosin: Nucleus: Median","Eosin: Nucleus: Min","Eosin: Nucleus: Max","Eosin: Nucleus: Std.Dev.",
    "Eosin: Cytoplasm: Mean","Eosin: Cytoplasm: Median","Eosin: Cytoplasm: Min","Eosin: Cytoplasm: Max","Eosin: Cytoplasm: Std.Dev.",
    "Eosin: Membrane: Mean","Eosin: Membrane: Median","Eosin: Membrane: Min","Eosin: Membrane: Max","Eosin: Membrane: Std.Dev.",
    "Eosin: Cell: Mean","Eosin: Cell: Median","Eosin: Cell: Min","Eosin: Cell: Max","Eosin: Cell: Std.Dev.",
]
# Two extra features appended at the end
EXTRA_FEATURES = ["Circumcircle", "polygon"]
OUTPUT_COLUMNS = ["cell_id","centroid_x","centroid_y","type"] + FEATURE_NAMES + EXTRA_FEATURES

# ------------------ Per-cell worker ------------------
def compute_cell_features_local(cid, cpoly, npoly, img_rgb, prob_map):
    H_img, W_img = img_rgb.shape[:2]

    # Compute bbox (with padding) that covers both cell and nucleus polygons
    polys = [p for p in [cpoly, npoly] if p is not None and not p.is_empty]
    if not polys:
        return None
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)

    xmin = max(int(np.floor(minx)) - BBOX_PAD, 0)
    ymin = max(int(np.floor(miny)) - BBOX_PAD, 0)
    xmax = min(int(np.ceil(maxx)) + BBOX_PAD, W_img)
    ymax = min(int(np.ceil(maxy)) + BBOX_PAD, H_img)
    if xmax <= xmin or ymax <= ymin:
        return None

    # Crop local RGB patch
    patch_rgb = img_rgb[ymin:ymax, xmin:xmax, :]

    # Rasterize polygons to local masks
    nuc_mask_local, _ = rasterize_polygon_local(npoly, H_img, W_img, bbox=(xmin, ymin, xmax, ymax))
    cell_mask_local, _ = rasterize_polygon_local(cpoly, H_img, W_img, bbox=(xmin, ymin, xmax, ymax))

    # If no cell polygon, try to expand nucleus to synthesize a cell mask
    if cpoly is None:
        if ENABLE_CELL_EXPAND_IF_MISSING and (nuc_mask_local.max() > 0):
            cell_mask_local = expand_nucleus_to_cell(nuc_mask_local, CELL_EXPANSION_FACTOR, CELL_CONSTRAIN_SCALE)
        else:
            if nuc_mask_local.max() == 0:
                return None

    # Constrain nucleus within cell
    nuc_mask_local = (nuc_mask_local & cell_mask_local).astype(np.uint8)

    # Local H&E deconvolution
    H_chan, E_chan = rgb_to_he_qupath_local(
        patch_rgb,
        p_low=LOCAL_P_LOW, p_high=LOCAL_P_HIGH,
        sample_step=LOCAL_PERCENTILE_SAMPLE_STEP
    )

    # Membrane and cytoplasm masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MEMBRANE_WIDTH+1, 2*MEMBRANE_WIDTH+1))
    dil = cv2.dilate(cell_mask_local, kernel, iterations=1)
    ero = cv2.erode(cell_mask_local, kernel, iterations=1)
    membrane_mask = ((dil - ero) > 0).astype(np.uint8)
    membrane_mask = (membrane_mask & cell_mask_local).astype(np.uint8)
    cytoplasm_mask = (cell_mask_local - nuc_mask_local).clip(0, 1).astype(np.uint8)

    # Centroid from polygon (global coordinates preferred)
    if cpoly is not None:
        cx, cy = cpoly.representative_point().coords[0]
        poly_for_geo = cpoly
    elif npoly is not None:
        cx, cy = npoly.representative_point().coords[0]
        poly_for_geo = npoly
    else:
        ys, xs = np.nonzero(cell_mask_local)
        if len(xs) == 0:
            return None
        cx = float(xmin + xs.mean())
        cy = float(ymin + ys.mean())
        poly_for_geo = None

    # Morphology from local masks
    n_area, n_perim, n_sol, n_maj, n_min, n_len, n_circ, n_maxd, n_mind = compute_morphology(nuc_mask_local)
    c_area, c_perim, c_sol, c_maj, c_min, c_len, c_circ, c_maxd, c_mind = compute_morphology(cell_mask_local)
    n_over_c_area = (n_area / c_area) if c_area > 0 else 0.0

    # Intensities
    H_n_vals = H_chan[nuc_mask_local == 1]
    H_cy_vals = H_chan[cytoplasm_mask == 1]
    H_m_vals  = H_chan[membrane_mask == 1]
    H_c_vals  = H_chan[cell_mask_local == 1]

    E_n_vals = E_chan[nuc_mask_local == 1]
    E_cy_vals = E_chan[cytoplasm_mask == 1]
    E_m_vals  = E_chan[membrane_mask == 1]
    E_c_vals  = E_chan[cell_mask_local == 1]

    Hn  = safe_stats(H_n_vals)
    Hcy = safe_stats(H_cy_vals)
    Hm  = safe_stats(H_m_vals)
    Hc  = safe_stats(H_c_vals)
    En  = safe_stats(E_n_vals)
    Ecy = safe_stats(E_cy_vals)
    Em  = safe_stats(E_m_vals)
    Ec  = safe_stats(E_c_vals)

    det_prob = float(prob_map.get(cid, 1.0))

    # New features: Circumcircle radius and polygon string (prefer cell polygon, fallback to nucleus)
    chosen_poly = cpoly if (cpoly is not None and not cpoly.is_empty) else (npoly if (npoly is not None and not npoly.is_empty) else None)
    circumcircle = polygon_radius_from_poly(chosen_poly)
    polygon_str = polygon_string_from_poly(chosen_poly)

    row = [
        str(cid),                      # cell_id as string
        float(cx), float(cy), "",      # type as empty string
        det_prob,
        float(cx), float(cy),
        n_area, n_len, n_circ, n_sol, n_maxd, n_mind, n_over_c_area,
        c_area, c_len, c_circ, c_sol, c_maxd, c_mind,
        Hn['mean'], Hn['median'], Hn['min'], Hn['max'], Hn['std'],
        Hcy['mean'], Hcy['median'], Hcy['min'], Hcy['max'], Hcy['std'],
        Hm['mean'], Hm['median'], Hm['min'], Hm['max'], Hm['std'],
        Hc['mean'], Hc['median'], Hc['min'], Hc['max'], Hc['std'],
        En['mean'], En['median'], En['min'], En['max'], En['std'],
        Ecy['mean'], Ecy['median'], Ecy['min'], Ecy['max'], Ecy['std'],
        Em['mean'], Em['median'], Em['min'], Em['max'], Em['std'],
        Ec['mean'], Ec['median'], Ec['min'], Ec['max'], Ec['std'],
        # Extra features
        circumcircle,
        polygon_str,
    ]
    return row

# ------------------ Main ------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if os.path.exists(PARQUET_OUT):
        os.remove(PARQUET_OUT)

    # Load image (whole image into memory; for very large WSIs consider tifffile memmap or tiled IO)
    print(f"Loading image: {IMAGE_PATH}")
    img = skio.imread(IMAGE_PATH)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] >= 3:
        img_rgb = img[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    H_img, W_img = img_rgb.shape[:2]

    # Load boundaries
    print("Loading boundaries ...")
    cell_df = pd.read_parquet(CELL_PARQUET)
    nuc_df = pd.read_parquet(NUC_PARQUET)

    # Map to image space using homography; note the SCALE normalization
    cell_df[['vertex_x', 'vertex_y']] = transform_points_inverse(cell_df[['vertex_x', 'vertex_y']].values / SCALE, H_ALIGN)
    nuc_df[['vertex_x', 'vertex_y']] = transform_points_inverse(nuc_df[['vertex_x', 'vertex_y']].values / SCALE, H_ALIGN)

    for need in ['cell_id', 'vertex_x', 'vertex_y']:
        if need not in cell_df.columns:
            raise ValueError(f"cell parquet missing column: {need}")
        if need not in nuc_df.columns:
            raise ValueError(f"nucleus parquet missing column: {need}")

    # Build shapely polygons
    print("Building polygons ...")
    cell_polys = {}
    nuc_polys = {}

    for cid, grp in tqdm(cell_df.groupby('cell_id'), desc="Cells"):
        poly = polygon_from_group(grp)
        if poly is None or poly.is_empty:
            continue
        cell_polys[cid] = poly

    for cid, grp in tqdm(nuc_df.groupby('cell_id'), desc="Nuclei"):
        poly = polygon_from_group(grp)
        if poly is None or poly.is_empty:
            continue
        nuc_polys[cid] = poly

    all_ids = sorted(set(nuc_polys.keys()) | set(cell_polys.keys()))
    print(f"Total unique IDs (union of cell & nucleus): {len(all_ids)}")

    # Optional detection probabilities
    prob_map = {}
    if PROB_PATH is not None and os.path.isfile(PROB_PATH):
        print(f"Loading detection probabilities: {PROB_PATH}")
        pdf = pd.read_csv(PROB_PATH, sep='\t|,', engine='python')
        cols_lower = {c: c.lower().replace(' ', '_') for c in pdf.columns}
        pdf = pdf.rename(columns=cols_lower)
        if 'cell_id' in pdf.columns and 'detection_probability' in pdf.columns:
            prob_map = dict(zip(pdf['cell_id'].astype(str), pdf['detection_probability']))
        elif 'cell_id' in pdf.columns and 'probability' in pdf.columns:
            prob_map = dict(zip(pdf['cell_id'].astype(str), pdf['probability']))
        elif 'cell_id' in pdf.columns and 'detection_prob' in pdf.columns:
            prob_map = dict(zip(pdf['cell_id'].astype(str), pdf['detection_prob']))
        else:
            warnings.warn("PROB_PATH provided but required columns not found; defaulting det_prob=1.0")

    # Parquet schema: polygon is string; numeric features are float64
    schema_fields = []
    for col in OUTPUT_COLUMNS:
        if col in ["cell_id", "type", "polygon"]:
            schema_fields.append(pa.field(col, pa.string()))
        else:
            schema_fields.append(pa.field(col, pa.float64()))
    schema = pa.schema(schema_fields)

    def rows_to_table(rows):
        cols = {c: [] for c in OUTPUT_COLUMNS}
        for r in rows:
            for c, v in zip(OUTPUT_COLUMNS, r):
                cols[c].append(v)
        arrays = []
        for c in OUTPUT_COLUMNS:
            if c in ["cell_id", "type", "polygon"]:
                arrays.append(pa.array([("" if v is None else str(v)) for v in cols[c]], type=pa.string()))
            else:
                arrays.append(pa.array(cols[c], type=pa.float64()))
        return pa.Table.from_arrays(arrays, names=OUTPUT_COLUMNS)

    writer = None
    print(f"Computing per-cell features locally with {NUM_WORKERS} threads ...")
    batch_rows = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {}
        for cid in all_ids:
            cpoly = cell_polys.get(cid, None)
            npoly = nuc_polys.get(cid, None)
            fut = ex.submit(
                compute_cell_features_local,
                str(cid), cpoly, npoly, img_rgb, prob_map
            )
            futures[fut] = cid

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Cells processed"):
            row = fut.result()
            if row is None:
                continue
            batch_rows.append(row)
            if len(batch_rows) >= BATCH_SIZE:
                table = rows_to_table(batch_rows)
                if writer is None:
                    writer = pq.ParquetWriter(PARQUET_OUT, table.schema, compression="zstd")
                writer.write_table(table)
                batch_rows.clear()

    # Flush tail
    if len(batch_rows) > 0:
        table = rows_to_table(batch_rows)
        if writer is None:
            writer = pq.ParquetWriter(PARQUET_OUT, table.schema, compression="zstd")
        writer.write_table(table)
        batch_rows.clear()

    if writer is not None:
        writer.close()

    print("Done. Saved to:", PARQUET_OUT)

if __name__ == "__main__":
    main()