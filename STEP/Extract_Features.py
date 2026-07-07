import argparse
import json
import os
import pickle
import sys

import anndata
import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from PIL import Image, ImageFile
from collections import defaultdict
from filelock import FileLock
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .utils import configure_logging  # noqa: E402  (local import after path tweak)


class SingleCellFeatures:
    """
    Orchestrates feature extraction for single-cell or spatial transcriptomics data.

    Depending on `hs_ST`, the pipeline either works directly on high-resolution segmentation
    (image mode) or reconstructs nuclei from CLAM predictions and GeoJSON polygons (sequence mode).
    """

    def __init__(self, tissue, out_dir, ST_Data, Img_Data, CLAM_Data, Json_Data, hs_ST):
        self.tissue = tissue
        self.base_out_dir = out_dir
        self.ST_Data = ST_Data
        self.Img_Data = Img_Data
        self.CLAM_Data = CLAM_Data
        self.Json_Data = Json_Data
        self.hs_ST = hs_ST

        self._ensure_output_dirs()
        self.loggings = configure_logging(os.path.join(self.out_dir, 'logs'))

        if hs_ST:
            self.LoadData_image(self.ST_Data, self.Img_Data)
        else:
            self.LoadData_seq(self.ST_Data, self.Img_Data, self.CLAM_Data, self.Json_Data)

    # ------------------------------------------------------------------
    # Geometry/statistics helpers
    # ------------------------------------------------------------------
    def calculate_area(self, points):
        """Return the number of pixels/points describing a nucleus."""
        return points.shape[0]

    def calculate_solidity(self, points):
        """Solidity = area / convex hull area."""
        convex_hull = Polygon(points[ConvexHull(points).vertices])
        return self.calculate_area(points) / convex_hull.area

    def calculate_max_diameter(self, points):
        """Maximum pairwise distance between hull vertices."""
        hull_points = points[ConvexHull(points).vertices]
        distances = np.array([np.linalg.norm(p1 - p2) for p1 in hull_points for p2 in hull_points])
        return np.max(distances)

    def calculate_min_diameter(self, points):
        """Minimum edge length along the convex hull perimeter."""
        hull_points = points[ConvexHull(points).vertices]
        distances = np.array([
            np.linalg.norm(hull_points[i] - hull_points[i - 1])
            for i in range(len(hull_points))
        ])
        return np.min(distances)

    def calculate_aspect_ratio(self, points):
        """Width-to-height ratio of the bounding box."""
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        return width / height

    def calculate_eccentricity(self, points):
        """Eccentricity derived from the min/max diameters."""
        min_diameter = self.calculate_min_diameter(points)
        max_diameter = self.calculate_max_diameter(points)
        return np.sqrt(1 - (min_diameter / max_diameter) ** 2)

    def calculate_equivalent_diameter(self, points):
        """Diameter of a circle with the same area as the nucleus."""
        area = self.calculate_area(points)
        return np.sqrt(4 * area / np.pi)

    def calculate_bounding_box_area(self, points):
        """Area of the bounding box tightly covering the nucleus."""
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        return width * height

    @staticmethod
    def center_point(polygon):
        """Return the box-center of a polygon defined as (y, x) tuples."""
        min_y = min(y for (y, x) in polygon)
        max_y = max(y for (y, x) in polygon)
        min_x = min(x for (y, x) in polygon)
        max_x = max(x for (y, x) in polygon)
        center_y = (min_y + max_y) / 2
        center_x = (min_x + max_x) / 2
        return center_y, center_x

    @staticmethod
    def reconstruct_mask(patch_coords, patch_size, mask_shape):
        """Rebuild a binary mask from patch coordinates."""
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for x, y in patch_coords:
            mask[y:y + patch_size, x:x + patch_size] = 1
        return mask

    @staticmethod
    def polygon_radius(polygon_points):
        """Compute the max distance from centroid to polygon edges (circumcircle radius)."""
        polygon = Polygon(polygon_points)
        center = np.array(polygon.centroid.xy).T[0]
        pts = np.asarray(polygon_points, dtype=float)
        return float(np.linalg.norm(pts - center, axis=1).max())

    @staticmethod
    def Convex(res, spot):
        """Filter cells that fall inside the convex hull of the ST spots."""
        hull_path = Path(spot[ConvexHull(spot).vertices])
        return res.iloc[hull_path.contains_points(res[['x', 'y']]), :]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def LoadData_seq(self, ST_Data, Img_Data, CLAM_Data, Json_Data):
        """Load ST AnnData, whole-slide image, CLAM patch coordinates, and GeoJSON cells."""
        self.loggings.info(f'Reading spatial data: {ST_Data}')
        sp_adata = self._read_spatial_data(ST_Data)
        self.sp_adata = sp_adata

        self.loggings.info(f'Reading image data: {Img_Data}')
        wsi = imageio.imread(Img_Data)
        self.width, self.height, _ = wsi.shape
        self.image = wsi
        self.loggings.info(f'Image shape: {(self.width, self.height)}')

        if CLAM_Data is not None:
            self.loggings.info(f'Reading CLAM data: {CLAM_Data}')
            with h5py.File(CLAM_Data, 'r') as f:
                self.patch_coords = f['coords'][:]
            self.patch_size = 256
            self.loggings.info(f'Number of patches: {self.patch_coords.shape[0]}')
            self.loggings.info(f'Patch size: {self.patch_size}*{self.patch_size}')
        else:
            # No CLAM tissue patches: skip mask-based filtering, keep all cells.
            self.patch_coords = None
            self.patch_size = 256
            self.loggings.info('No CLAM data provided; tissue mask filtering disabled.')

        self.loggings.info(f'Reading Json data: {Json_Data}')
        with open(Json_Data, 'r') as file:
            self.geojson = json.load(file)
        self.loggings.info(f'Number of cells: {len(self.geojson["features"]) - 1}')

    def LoadData_image(self, ST_Data, Img_Data):
        """Load ST AnnData along with precomputed per-cell contours."""
        self.loggings.info(f'Reading spatial data: {ST_Data}')
        sp_adata = self._read_spatial_data(ST_Data)
        self.sp_adata = sp_adata

        self.loggings.info(f'Reading image data: {Img_Data}')
        with open(Img_Data, 'rb') as f:
            self.image = pickle.load(f)
        self.loggings.info(f'Number of all cells: {len(self.image)}')

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def ExtractFeatures(self, process_log, part=False):
        """
        Extract cellular morphometrics and store them inside `self.sp_adata.uns['features']`.
        Progress is tracked through a file specified by `process_log`.
        """
        if not self.hs_ST:
            if self.patch_coords is not None:
                mask_shape = (self.width, self.height)
                mask = self.reconstruct_mask(self.patch_coords, self.patch_size, mask_shape)
            else:
                # Without CLAM patches, accept every cell (mask is all ones).
                mask = None

            features_template = list(self.geojson['features'][1]['properties']['measurements'].keys())
            feats = self.geojson['features']
            total = len(feats)

            # Accumulate rows in plain lists and build the DataFrame once at the end;
            # per-row `res.loc[...]` assignment is O(n) each and dominates runtime.
            records, xs, ys, circ, polys = [], [], [], [], []
            progress_step = max(1, total // 100)

            for index in tqdm(range(total)):
                # Update progress (0–95%) periodically to avoid per-cell lock overhead.
                if index % progress_step == 0 or index == total - 1:
                    with FileLock(process_log + '.lock'):
                        with open(process_log, 'w') as f:
                            f.write(str(int((index + 1) / total * 95)))

                props = feats[index].get('properties', {})
                if 'measurements' not in props:
                    continue
                try:
                    feat = feats[index]
                    polygon = [(i, j) for i, j in np.array(feat['geometry']['coordinates'][0]).squeeze()]
                    center_x, center_y = self.center_point(polygon)

                    if mask is None or mask[round(center_y), round(center_x)] == 1:
                        measurements = props['measurements']
                        records.append([measurements.get(k) for k in features_template])
                        xs.append(center_x)
                        ys.append(center_y)
                        circ.append(self.polygon_radius(polygon))
                        polys.append(str(polygon))
                except Exception:
                    # Silently skip malformed entries to mimic original behavior.
                    pass

            res = pd.DataFrame(records, columns=features_template)
            res['x'] = xs
            res['y'] = ys
            res['Circumcircle'] = circ
            res['polygon'] = polys

            if part:
                # Optionally keep only cells within the convex hull of the ST spots.
                res = self.Convex(res, self.sp_adata.obsm['spatial'])

            res.index = [f'cell_{i}' for i in range(res.shape[0])]
            polygon_backup = res['polygon'].copy()
            del res['polygon']

            res = res.astype(np.float32)
            res = res.apply(lambda x: x.fillna(x.mean()), axis=0)
            res['polygon'] = polygon_backup

            self.sp_adata.uns['features'] = res
            self.loggings.info(f"Number of cells in tissue: {res.shape[0]}")

            # Visualization panel (H&E, tissue mask, spots, cells).
            fig, axes = plt.subplots(1, 4, figsize=(30, 9), dpi=250)
            axes[0].imshow(self.image)
            axes[0].set_title("H&E")

            axes[1].imshow(self.image if mask is None else self.image * mask[:, :, None])
            axes[1].set_title("Tissue Profile")

            axes[2].scatter(self.sp_adata.obsm['spatial'][:, 0], self.sp_adata.obsm['spatial'][:, 1], s=5)
            axes[2].set_title("Spots")
            axes[2].invert_yaxis()

            axes[3].scatter(res['x'], res['y'], s=0.1)
            axes[3].set_title("Cells")
            axes[3].invert_yaxis()

            plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
            plt.close()

        else:
            # Directly use per-cell polygons/point clouds supplied in `self.image`.
            features = pd.DataFrame(columns=[
                'area', 'solidity', 'max_diameter', 'min_diameter',
                'aspect_ratio', 'eccentricity', 'equivalent_diameter', 'bounding_box_area'
            ])

            for cell, points in self.image.items():
                if not np.isin(cell, self.sp_adata.obs.index.values):
                    continue

                plt.scatter(points[:, 0], points[:, 1], s=0.01)

                area = self.calculate_area(points)
                solidity = self.calculate_solidity(points)
                max_diameter = self.calculate_max_diameter(points)
                min_diameter = self.calculate_min_diameter(points)
                aspect_ratio = self.calculate_aspect_ratio(points)
                eccentricity = self.calculate_eccentricity(points)
                equivalent_diameter = self.calculate_equivalent_diameter(points)
                bounding_box_area = self.calculate_bounding_box_area(points)

                features.loc[cell, :] = [
                    area, solidity, max_diameter, min_diameter,
                    aspect_ratio, eccentricity, equivalent_diameter, bounding_box_area
                ]

            features = features.astype(np.float32)
            features = features.apply(lambda x: x.fillna(x.mean()), axis=0)
            self.sp_adata.uns['features'] = features

            plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
            plt.close()

        # Persist results and mark completion.
        self.sp_adata.write_h5ad(os.path.join(self.out_dir, 'sp_adata_ef.h5ad'))
        with FileLock(process_log + '.lock'):
            with open(process_log, 'w') as f:
                f.write('100')

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _ensure_output_dirs(self):
        """Create base/tissue-level directories if they do not exist."""
        os.makedirs(self.base_out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_out_dir, self.tissue), exist_ok=True)
        self.out_dir = os.path.join(self.base_out_dir, self.tissue)

    @staticmethod
    def _read_spatial_data(ST_Data):
        """Load an AnnData object and ensure unique obs/var names."""
        sp_adata = anndata.read_h5ad(ST_Data)
        sp_adata.obs.index = sp_adata.obs.index.astype('object')
        sp_adata.obs_names_make_unique()
        sp_adata.var.index = sp_adata.var.index.astype('object')
        sp_adata.var_names_make_unique()
        return sp_adata