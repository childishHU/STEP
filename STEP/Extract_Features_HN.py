data_path = '/data1/hzq/idea/TGCA/PDAC/slice'
json_path = '/data1/hzq/idea/TGCA/PDAC/json'
save_path = '/data1/hzq/idea/TGCA/PDAC/Features'


import cv2
import numpy as np
import json
from openslide import open_slide, ImageSlide
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from skimage.measure import regionprops, label, regionprops_table
import math
import random
import pickle
import os
from multiprocessing import Pool
import shutil
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from skimage import io, color
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
import psutil

'''
feature_list = ["area", "perimeter", "orientation", "eccentricity", "solidity", "axis_major_length", "axis_minor_length",
 "nucleus: mean", "nucleus: median", "nucleus: min", "nucleus: max", "nucleus: std", "nucleus: skew", "nucleus: kurtosis",
 "cell: mean", "cell: median", "cell: min", "cell: max", "cell: std", "cell: skew", "cell: kurtosis",
 "nucleus cell ratio",
 "nucleus H: mean", "nucleus H: median", "nucleus H: min", "nucleus H: max", "nucleus H: std", "nucleus H: skew", "nucleus H: kurtosis",
 "cytoplasm H: mean", "cytoplasm H: median", "cytoplasm H: min", "cytoplasm H: max", "cytoplasm H: std", "cytoplasm H: skew", "cytoplasm H: kurtosis",
 "membrane H: mean", "membrane H: median", "membrane H: min", "membrane H: max", "membrane H: std", "membrane H: skew", "membrane H: kurtosis",
 "cell H: mean", "cell H: median", "cell H: min", "cell H: max", "cell H: std", "cell H: skew", "cell H: kurtosis",
 "nucleus E: mean", "nucleus E: median", "nucleus E: min", "nucleus E: max", "nucleus E: std", "nucleus E: skew", "nucleus E: kurtosis",
 "cytoplasm E: mean", "cytoplasm E: median", "cytoplasm E: min", "cytoplasm E: max", "cytoplasm E: std", "cytoplasm E: skew", "cytoplasm E: kurtosis",
 "membrane E: mean", "membrane E: median", "membrane E: min", "membrane E: max", "membrane E: std", "membrane E: skew", "membrane E: kurtosis",
 "cell E: mean", "cell E: median", "cell E: min", "cell E: max", "cell E: std", "cell E: skew", "cell E: kurtosis",
]
'''
SUPPORTED_EXTS = (".svs", ".tif", ".tiff")  
NUM_WORKERS = 10
hovernet_mag = 40


if not os.path.isdir(save_path):
    os.mkdir(save_path)

final_list = []

for file in os.listdir(json_path):
    if 'json' in file:
        final_list.append(file.split('/')[-1][:-4])

print(len(final_list))


def rgb_to_he(rgb_image):
    stain_matrix = np.array([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78]
    ])

  
    od_image = -np.log(rgb_image / 255.0 + 1e-6)  

    stain_concentrations = np.linalg.lstsq(stain_matrix.T, od_image.reshape(-1, 3).T, rcond=None)[0]
    stain_concentrations = stain_concentrations.reshape(3, *rgb_image.shape[:2])

  
    h_channel = stain_concentrations[0]
    e_channel = stain_concentrations[1] 


    h_channel = rescale_intensity(h_channel, out_range=(0, 1))
    e_channel = rescale_intensity(e_channel, out_range=(0, 1))
    return np.uint8(255 * h_channel), np.uint8(255 * e_channel)

def find_slide(data_path, file_stem):

    p = os.path.join(data_path, file_stem)
    if os.path.isfile(p):
        return os.path.abspath(p)

    stem, ext = os.path.splitext(file_stem)
    candidates = []
    if ext:
        candidates.append(file_stem)
        for e in SUPPORTED_EXTS:
            candidates.append(stem + e)
    else:
        for e in SUPPORTED_EXTS:
            candidates.append(file_stem + e)

    for name in candidates:
        p = os.path.join(data_path, name)
        if os.path.isfile(p):
            return os.path.abspath(p)
        
    for sub in os.listdir(data_path):
        subdir = os.path.join(data_path, sub)
        if not os.path.isdir(subdir):
            continue
        for name in candidates:
            p = os.path.join(subdir, name)
            if os.path.isfile(p):
                return os.path.abspath(p)

    return None

def run_extraction(file_name):
    svs_file_path = find_slide(data_path, file_name)

    json_file_path = json_path + '/' + file_name + 'json'
    save_pickle_file_path = save_path + '/' + file_name + 'pickle'
    if svs_file_path is not None:
        if not os.path.isfile(save_pickle_file_path):
            # if True:
            print(file_name)

            slide = open_slide(svs_file_path)
            if 'aperio.AppMag' in slide.properties:
                slide_mag = int(slide.properties['aperio.AppMag'][:2])
            else:
                slide_mag = 1
            mag_ratio = hovernet_mag / slide_mag

            with open(json_file_path) as f:
                pred_data = json.load(f)

            if len(list(pred_data['nuc'].keys())) == 0:
                print('No nuclei in the image, skip')

            elif len(list(pred_data['nuc'].keys())) >= 1:
                prop_dict = {}
                all_keys = list(pred_data['nuc'].keys())

                if len(all_keys) > 100000:
                    sampled_keys = random.sample(all_keys, int(len(all_keys) * 0.1))
                else:
                    sampled_keys = all_keys
                for keys in tqdm(sampled_keys):
                    # for keys in list(pred_data['nuc'].keys())[:10]:
                    max_x, max_y = slide.level_dimensions[0]
                    temp_contour = np.array(pred_data['nuc'][keys]['contour']).copy() // mag_ratio

                    x_min = int(temp_contour[:, 0].min()) - 10
                    if x_min < 0:
                        x_min = 0
                    x_max = int(temp_contour[:, 0].max()) + 10
                    if x_max > max_x:
                        x_max = max_x - 1
                    y_min = int(temp_contour[:, 1].min()) - 10
                    if y_min < 0:
                        y_min = 0
                    y_max = int(temp_contour[:, 1].max()) + 10
                    if y_max > max_y:
                        y_max = max_y - 1

                    wsi_cell_crop = slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min))
                    wsi_cell_crop_color = np.array(wsi_cell_crop)[:, :, :3]

                    cell_image = Image.new("RGB", (x_max - x_min, y_max - y_min))

                    draw = ImageDraw.Draw(cell_image)
                    
                    temp_contour[:, 0] = temp_contour[:, 0] - x_min
                    temp_contour[:, 1] = temp_contour[:, 1] - y_min
                    
                    draw.polygon((tuple(map(tuple, temp_contour))), fill=(1))

                    cell_image = np.array(cell_image)[:, :, 0] # nucleus
                    kernel = np.ones((5, 5), np.uint8)
                    cu = cv2.dilate(cell_image, kernel, iterations=1) 
                    cell = cv2.dilate(cu, kernel, iterations=1) # cell
                    cytoplasm = cell - cell_image
                    membrane = cell - cu
                    # shape
                    properties_list = ["area", "perimeter", "orientation", "eccentricity", "solidity",
                                        "axis_major_length", "axis_minor_length"]

                    cell_property = regionprops_table(cell_image, intensity_image=wsi_cell_crop_color,
                                                        properties=properties_list)

                    temp_prop_list = np.array(list(cell_property.values())).reshape(-1).tolist()

                    wsi_cell_crop = (rgb2gray(wsi_cell_crop_color) * 255).astype(np.uint8)

                    
                    
                    # nucleus features
                    wsi_cell_crop_masked = wsi_cell_crop[np.where(cell_image == 1)[0], np.where(cell_image == 1)[1]].copy()
                    temp_prop_list.append(wsi_cell_crop_masked.mean())
                    temp_prop_list.append(np.median(wsi_cell_crop_masked))
                    temp_prop_list.append(wsi_cell_crop_masked.min())
                    temp_prop_list.append(wsi_cell_crop_masked.max())
                    temp_prop_list.append(wsi_cell_crop_masked.std())
                    temp_prop_list.append(skew(wsi_cell_crop_masked))
                    temp_prop_list.append(kurtosis(wsi_cell_crop_masked))

                    # cell features
                    wsi_cell_crop_masked = wsi_cell_crop[np.where(cell == 1)[0], np.where(cell == 1)[1]].copy()
                    temp_prop_list.append(wsi_cell_crop_masked.mean())
                    temp_prop_list.append(np.median(wsi_cell_crop_masked))
                    temp_prop_list.append(wsi_cell_crop_masked.min())
                    temp_prop_list.append(wsi_cell_crop_masked.max())
                    temp_prop_list.append(wsi_cell_crop_masked.std())
                    temp_prop_list.append(skew(wsi_cell_crop_masked))
                    temp_prop_list.append(kurtosis(wsi_cell_crop_masked))
                    
                    # nucleus / cell
                    temp_prop_list.append(np.sum(cell_image) / np.sum(cell))
                    
                    # H E channel:
                    H, E = rgb_to_he(wsi_cell_crop_color.copy())

                    # H nucleus
                    H_crop_masked = H[np.where(cell_image == 1)[0], np.where(cell_image == 1)[1]].copy()
                    temp_prop_list.append(H_crop_masked.mean())
                    temp_prop_list.append(np.median(H_crop_masked))
                    temp_prop_list.append(H_crop_masked.min())
                    temp_prop_list.append(H_crop_masked.max())
                    temp_prop_list.append(H_crop_masked.std())
                    temp_prop_list.append(skew(H_crop_masked))
                    temp_prop_list.append(kurtosis(H_crop_masked))
                    # H cytoplasm
                    H_crop_masked = H[np.where(cytoplasm == 1)[0], np.where(cytoplasm == 1)[1]].copy()
                    temp_prop_list.append(H_crop_masked.mean())
                    temp_prop_list.append(np.median(H_crop_masked))
                    temp_prop_list.append(H_crop_masked.min())
                    temp_prop_list.append(H_crop_masked.max())
                    temp_prop_list.append(H_crop_masked.std())
                    temp_prop_list.append(skew(H_crop_masked))
                    temp_prop_list.append(kurtosis(H_crop_masked))
                    # H membrane
                    H_crop_masked = H[np.where(membrane == 1)[0], np.where(membrane == 1)[1]].copy()
                    temp_prop_list.append(H_crop_masked.mean())
                    temp_prop_list.append(np.median(H_crop_masked))
                    temp_prop_list.append(H_crop_masked.min())
                    temp_prop_list.append(H_crop_masked.max())
                    temp_prop_list.append(H_crop_masked.std())
                    temp_prop_list.append(skew(H_crop_masked))
                    temp_prop_list.append(kurtosis(H_crop_masked))
                    # H cell
                    H_crop_masked = H[np.where(cell == 1)[0], np.where(cell == 1)[1]].copy()
                    temp_prop_list.append(H_crop_masked.mean())
                    temp_prop_list.append(np.median(H_crop_masked))
                    temp_prop_list.append(H_crop_masked.min())
                    temp_prop_list.append(H_crop_masked.max())
                    temp_prop_list.append(H_crop_masked.std())
                    temp_prop_list.append(skew(H_crop_masked))
                    temp_prop_list.append(kurtosis(H_crop_masked))
                    # E nucleus
                    E_crop_masked = E[np.where(cell_image == 1)[0], np.where(cell_image == 1)[1]].copy()
                    temp_prop_list.append(E_crop_masked.mean())
                    temp_prop_list.append(np.median(E_crop_masked))
                    temp_prop_list.append(E_crop_masked.min())
                    temp_prop_list.append(E_crop_masked.max())
                    temp_prop_list.append(E_crop_masked.std())
                    temp_prop_list.append(skew(E_crop_masked))
                    temp_prop_list.append(kurtosis(E_crop_masked))
                    # E cytoplasm
                    E_crop_masked = E[np.where(cytoplasm == 1)[0], np.where(cytoplasm == 1)[1]].copy()
                    temp_prop_list.append(E_crop_masked.mean())
                    temp_prop_list.append(np.median(E_crop_masked))
                    temp_prop_list.append(E_crop_masked.min())
                    temp_prop_list.append(E_crop_masked.max())
                    temp_prop_list.append(E_crop_masked.std())
                    temp_prop_list.append(skew(E_crop_masked))
                    temp_prop_list.append(kurtosis(E_crop_masked))
                    # E membrane
                    E_crop_masked = E[np.where(membrane == 1)[0], np.where(membrane == 1)[1]].copy()
                    temp_prop_list.append(E_crop_masked.mean())
                    temp_prop_list.append(np.median(E_crop_masked))
                    temp_prop_list.append(E_crop_masked.min())
                    temp_prop_list.append(E_crop_masked.max())
                    temp_prop_list.append(E_crop_masked.std())
                    temp_prop_list.append(skew(E_crop_masked))
                    temp_prop_list.append(kurtosis(E_crop_masked))
                    # E cell
                    E_crop_masked = E[np.where(cell == 1)[0], np.where(cell == 1)[1]].copy()
                    temp_prop_list.append(E_crop_masked.mean())
                    temp_prop_list.append(np.median(E_crop_masked))
                    temp_prop_list.append(E_crop_masked.min())
                    temp_prop_list.append(E_crop_masked.max())
                    temp_prop_list.append(E_crop_masked.std())
                    temp_prop_list.append(skew(E_crop_masked))
                    temp_prop_list.append(kurtosis(E_crop_masked))
                    # texture features
                    # glcm = graycomatrix(wsi_cell_crop * cell_image,
                    #                    distances=[1], angles=[0], levels=256)

                    # temp_prop_list.extend([graycoprops(glcm, 'contrast')[0][0], graycoprops(glcm, 'dissimilarity')[0][0],
                    #                       graycoprops(glcm, 'homogeneity')[0][0], graycoprops(glcm, 'energy')[0][0]])

                    poly_level0 = np.array(pred_data['nuc'][keys]['contour']).copy() / mag_ratio
                    poly_crop = poly_level0.copy()
                    poly_crop[:, 0] -= x_min
                    poly_crop[:, 1] -= y_min

                    prop_dict[keys] = {}
                    prop_dict[keys]['centroid'] = np.array(pred_data['nuc'][keys]['centroid']) // mag_ratio
                    prop_dict[keys]['type'] = pred_data['nuc'][keys]['type']
                    prop_dict[keys]['properties'] = temp_prop_list

                    prop_dict[keys]['poly_level0'] = poly_level0.tolist()  
                    prop_dict[keys]['poly_crop'] = poly_crop.tolist()

                with open(save_pickle_file_path, 'wb') as f:
                    pickle.dump(prop_dict, f)

                del prop_dict
                del pred_data

            

    else:
        print('error loading: ' + file_name)


def prepare_and_save(file):
    run_extraction(file)


p = Pool(NUM_WORKERS)
print(p.map(prepare_and_save, final_list))