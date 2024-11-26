import numpy as np
import imageio.v2 as imageio
import json
import pandas as pd
import h5py
from shapely.geometry import Polygon, Point
import argparse
import sys
import anndata
import os
import pickle
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import configure_logging
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class SingleCellFeatures:
    def __init__(self,tissue,out_dir,ST_Data,Img_Data,CLAM_Data,Json_Data,max_cell_number,hs_ST):
        self.tissue = tissue
        self.out_dir = out_dir 
        self.ST_Data = ST_Data
        self.Img_Data = Img_Data
        self.CLAM_Data = CLAM_Data
        self.Json_Data = Json_Data
        self.max_cell_number = max_cell_number
        self.hs_ST = hs_ST
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(os.path.join(out_dir,tissue)):
            os.mkdir(os.path.join(out_dir,tissue))
        
        self.out_dir = os.path.join(out_dir,tissue)
        loggings = configure_logging(os.path.join(self.out_dir,'logs'))
        self.loggings = loggings 
        if hs_ST:
            self.LoadData_image(self.ST_Data, self.Img_Data)
        else:
            self.LoadData_seq(self.ST_Data, self.Img_Data, self.CLAM_Data, self.Json_Data)

    def calculate_area(self, points):
        return points.shape[0]

    def calculate_solidity(self, points):
        convex_hull = Polygon(points[ConvexHull(points).vertices])
        return self.calculate_area(points) / convex_hull.area

    def calculate_max_diameter(self, points):
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        distances = np.array([np.linalg.norm(p1 - p2) for p1 in hull_points for p2 in hull_points])
        return np.max(distances)

    def calculate_min_diameter(self, points):
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        distances = np.array([np.linalg.norm(hull_points[i] - hull_points[i-1]) for i in range(len(hull_points))])
        return np.min(distances)

    def calculate_aspect_ratio(self, points):
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        return width / height

    def calculate_eccentricity(self, points):
        min_diameter = self.calculate_min_diameter(points)
        max_diameter = self.calculate_max_diameter(points)
        return np.sqrt(1 - (min_diameter / max_diameter) ** 2)

    def calculate_equivalent_diameter(self, points):
        area = self.calculate_area(points)
        return np.sqrt(4 * area / np.pi)

    def calculate_bounding_box_area(self, points):
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        return width * height


    @staticmethod
    def center_point(polygon):
        min_y = min([y for (y, x) in polygon])
        max_y = max([y for (y, x) in polygon])
        min_x = min([x for (y, x) in polygon])
        max_x = max([x for (y, x) in polygon])
        center_y = (min_y + max_y) / 2  # h
        center_x = (min_x + max_x) / 2  # w
        return center_y, center_x
    
    @staticmethod
    def reconstruct_mask(patch_coords, patch_size, mask_shape):
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for coords in patch_coords:
            x, y = coords  
            mask[y:y+patch_size, x:x+patch_size] = 1
        return mask
    
    @staticmethod
    def polygon_radius(polygon_points):
        polygon = Polygon(polygon_points)
        center = np.array(polygon.centroid.xy).T[0]
        max_distance = 0
        for point in polygon_points:
            distance = np.linalg.norm(np.array(point) - center)
            max_distance = max(max_distance, distance)
        return max_distance
    
    @staticmethod
    def Convex(res, spot):
        hull = ConvexHull(spot)
        hull_path = Path(spot[hull.vertices])
        return res.iloc[hull_path.contains_points(res[['x','y']]), :]
    
    def LoadData_seq(self, ST_Data, Img_Data, CLAM_Data, Json_Data):
        self.loggings.info(f'Reading spatial data: {ST_Data}')
        sp_adata = anndata.read_h5ad(ST_Data)
        sp_adata.obs.index = sp_adata.obs.index.astype('object')
        sp_adata.obs_names_make_unique()
        sp_adata.var.index = sp_adata.var.index.astype('object')
        sp_adata.var_names_make_unique()
        #sp_adata.obsm['spatial'] = sp_adata.obsm['spatial'].to_numpy()
        self.loggings.info(f'Spatial data shape: {sp_adata.shape}')
        self.sp_adata = sp_adata
        self.loggings.info(f'Reading image data: {Img_Data}')
        wsi = imageio.imread(Img_Data)
        self.width, self.height, _ = wsi.shape
        self.image = wsi
        self.loggings.info(f'Image shape: {self.width, self.height}')
        self.loggings.info(f'Reading CLAM data: {CLAM_Data}')
        f = h5py.File(CLAM_Data, 'r')
        self.patch_coords = f['coords'][:]
        self.patch_size = 256
        self.loggings.info(f'Number of patches: {self.patch_coords.shape[0]}')
        self.loggings.info(f'Patch size: {self.patch_size}*{self.patch_size}')
        self.loggings.info(f'Reading Json data: {Json_Data}')
        with open(Json_Data, 'r') as file:
            self.geojson = json.load(file)
        self.loggings.info(f'Number of cells: {len(self.geojson["features"]) - 1}')
    

    def LoadData_image(self, ST_Data, Img_Data):
        self.loggings.info(f'Reading spatial data: {ST_Data}')
        sp_adata = anndata.read_h5ad(ST_Data)
        sp_adata.obs.index = sp_adata.obs.index.astype('object')
        sp_adata.obs_names_make_unique()
        sp_adata.var.index = sp_adata.var.index.astype('object')
        sp_adata.var_names_make_unique()
        #sp_adata.obsm['spatial'] = sp_adata.obsm['spatial'].to_numpy()
        self.loggings.info(f'Spatial data shape: {sp_adata.shape}')
        self.sp_adata = sp_adata
        self.loggings.info(f'Reading image data: {Img_Data}')
        with open(Img_Data, 'rb') as f:
            self.image = pickle.load(f)
        self.loggings.info(f'Number of all cells: {len(self.image)}')
        
    
    def ExtractFeatures(self, part=False):
        if not self.hs_ST:
            mask_shape = (self.width, self.height) 
            mask = self.reconstruct_mask(self.patch_coords, self.patch_size, mask_shape)
            length = len(self.geojson['features'])
            res = pd.DataFrame(columns=self.geojson['features'][1]['properties']['measurements'].keys())
            for index in range(length):
                if 'measurements' in self.geojson['features'][index]['properties']:
                    features = self.geojson['features'][index] 
                    polygon = [(i, j) for i, j in np.array(features['geometry']['coordinates'][0]).squeeze()]
                    center_x, center_y = self.center_point(polygon) 
                    if mask[round(center_y), round(center_x)] == 1:
                        res.loc[index] = features['properties']['measurements']
                        res.loc[index, ['x','y']] = center_x, center_y
                        res.loc[index, 'Circumcircle'] = self.polygon_radius(polygon)
                        res.loc[index, 'polygon'] = str(polygon)


            if part:
                res = self.Convex(res, self.sp_adata.obsm['spatial'])

            res.index = ['cell_'+str(i) for i in range(res.shape[0])]
            polygon = res['polygon'].copy()
            del res['polygon']
            res = res.astype(np.float32)
            res = res.apply(lambda x: x.fillna(x.mean()), axis=0)
            res['polygon'] = polygon

            self.sp_adata.uns['features'] = res
            self.loggings.info(f"Number of cells in tissue: {res.shape[0]}")
            fig, axes = plt.subplots(1, 4,figsize=(30,9),dpi=250)
            axes[0].imshow(self.image)
            axes[0].set_title("H&E")
            axes[1].imshow(self.image * mask[:,:,None])
            axes[1].set_title("Tissue Profile")
            axes[2].scatter(self.sp_adata.obsm['spatial'][:,0], self.sp_adata.obsm['spatial'][:,1], s = 5)
            axes[2].set_title("Spots")
            axes[2].invert_yaxis()
            axes[3].scatter(res['x'], res['y'], s=0.1)
            axes[3].set_title("Cells")
            axes[3].invert_yaxis()
            plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
            plt.close()

        else:
            features = pd.DataFrame(columns=['area', 'solidity', 'max_diameter', 'min_diameter', 'aspect_ratio', 'eccentricity', 'equivalent_diameter', 'bounding_box_area'])
            for cell in self.image.keys():
                if np.isin(cell, self.sp_adata.obs.index.values):
                    points = self.image[cell]
                    plt.scatter(points[:,0], points[:,1],s=0.01)
                    area = self.calculate_area(points)
                    solidity = self.calculate_solidity(points)
                    max_diameter = self.calculate_max_diameter(points)
                    min_diameter = self.calculate_min_diameter(points)
                    aspect_ratio = self.calculate_aspect_ratio(points)
                    eccentricity = self.calculate_eccentricity(points)
                    equivalent_diameter = self.calculate_equivalent_diameter(points)
                    bounding_box_area = self.calculate_bounding_box_area(points)    
                    features.loc[cell, :] = [area, solidity, max_diameter, min_diameter, aspect_ratio, eccentricity, equivalent_diameter, bounding_box_area]
            features = features.astype(np.float32)
            features = features.apply(lambda x: x.fillna(x.mean()), axis=0)
            self.sp_adata.uns['features'] = features
            plt.savefig(os.path.join(self.out_dir, 'nuclei_segmentation.png'))
            plt.close()

        self.sp_adata.write_h5ad(os.path.join(self.out_dir, 'sp_adata_ef.h5ad'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='simulation sour_sep')
    parser.add_argument('--tissue', type=str, help='tissue name', default=None)
    parser.add_argument('--out_dir', type=str, help='output path', default=None)
    parser.add_argument('--ST_Data', type=str, help='ST data path', default=None)
    parser.add_argument('--Img_Data', type=str, help='H&E stained image data path', default=None)
    parser.add_argument('--CLAM_Data', type=str, help='CLAM data(pathes) path', default=None)
    parser.add_argument('--Json_Data', type=str, help='Json data(from QuPath & StarDist) path', default=None)
    parser.add_argument('--max_cell_number', type=int, help='maximum cell number per spot', default=20)
    parser.add_argument('--part', type=bool, help='Are all spots distributed on issue?', default=False)
    parser.add_argument('--hs_ST', type=bool, help='image-based or seq-based', default=False)
    args = parser.parse_args()
    
    SCF = SingleCellFeatures(args.tissue, args.out_dir, args.ST_Data, args.Img_Data, args.CLAM_Data, args.Json_Data,args.max_cell_number,args.hs_ST)
    SCF.ExtractFeatures(args.part)