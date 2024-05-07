# calculate mean and std deviation ovar aall tiles

import glob, os
import numpy as np
from osgeo import gdal

# calculate img statistics
def img_stats(base_path):
    files = glob.glob(os.path.join(base_path, '*.png'))

    n_tiles = len(files)
    print("tiles:", n_tiles)
    n_bands = gdal.Open(files[0]).ReadAsArray().shape[0]
    print("bands:", n_bands)

    mean = np.zeros(n_bands)
    stdTemp = np.zeros(n_bands)
    std = np.zeros(n_bands)

    for t in range(0, n_tiles):
        im = gdal.Open(files[t]).ReadAsArray()
        n_bands = im.shape[0]
        for b in range(0, n_bands):
            mean[b] += np.nanmean(im[b,:,:])
    mean = (mean/n_tiles)
    print("mean:", mean)


    for t in range(0, n_tiles):
        im = gdal.Open(files[t]).ReadAsArray()
        n_bands = im.shape[0]
        for b in range(0, n_bands):
            stdTemp[b] += np.nansum((im[b,:,:] - mean[b])**2)/(im.shape[1]*im.shape[2])
            
    std = np.sqrt(stdTemp/n_tiles)
    print("std:", std)

    return n_bands, mean, std

# normaliser tensor mellom 0 og 1
def normalize(tensor):
        tensor -= tensor.flatten().min(0)[0]
        tensor /= tensor.flatten().max(0)[0]
        return tensor

# calculate class weight
def weights(base_path):
    files = glob.glob(os.path.join(base_path, '*.png'))
    n_tiles = len(files)
    n_buildings = 0
    n_roads = 0
    n_vegetation = 0
    n_water = 0
    n_open = 0

    for t in range(n_tiles):
        im = gdal.Open(files[t]).ReadAsArray()
        unique_values, counts = np.unique(im, return_counts=True)
        if 0 in unique_values:
            n_buildings += counts[np.where(unique_values == 0)]
        else:
            n_buildings += 0
        if 1 in unique_values:
            n_roads += counts[np.where(unique_values == 1)]
        else:
            n_roads += 0
        if 2 in unique_values:
            n_vegetation += counts[np.where(unique_values == 2)]
        else:
            n_vegetation += 0
        if 3 in unique_values:
            n_water += counts[np.where(unique_values == 3)]
        else:
            n_water += 0
        if 4 in unique_values:
            n_open += counts[np.where(unique_values == 4)]
        else:
            n_open += 0

    buildings_weight = 1-(n_buildings/(n_buildings+n_roads+n_vegetation+n_water+n_open))
    roads_weight = 1-(n_roads/(n_buildings+n_roads+n_vegetation+n_water+n_open))
    vegetation_weight = 1-(n_vegetation/(n_buildings+n_roads+n_vegetation+n_water+n_open))
    water_weight = 1-(n_water/(n_buildings+n_roads+n_vegetation+n_water+n_open))
    open_weight = 1-(n_open/(n_buildings+n_roads+n_vegetation+n_water+n_open))
    return [buildings_weight, roads_weight, vegetation_weight,water_weight,open_weight]



# show a image
import rasterio
from rasterio.plot import show, show_hist
import warnings
warnings.filterwarnings("ignore")

def pct_clip(array, pct=[2,98]):
        array_min, array_max = np.nanpercentile(array, pct[0]), np.nanpercentile(array, pct[1])
        clip = (array - array_min) / (array_max - array_min)
        clip[clip>1] = 1
        clip[clip<0] = 0
        return clip


def show_msi(path, bands=[4,3,2], pct=[2,98], hist=False, meta=False):
    raster = rasterio.open(path)
    
    r = pct_clip(raster.read(bands[0]), pct=pct)
    g = pct_clip(raster.read(bands[1]), pct=pct)
    b = pct_clip(raster.read(bands[2]), pct=pct)

    if meta:
        print(raster.meta)
    if hist:
        show_hist(source=raster, bins=50, title='Histogram',
                histtype='stepfilled', alpha=0.5)
    
    show(np.array([r,g,b]))
    

def show_sar(path, pct=[2,98], hist=False, meta=False):

    raster = rasterio.open(path)
    if meta:
        print(raster.meta)
    if hist:
        show_hist(source=raster, bins=50, title='Histogram',
                histtype='stepfilled', alpha=0.5)
    
    raster_arr_1 = pct_clip(raster.read(1), pct=pct)
    raster_arr_2 = pct_clip(raster.read(2), pct=pct)

    r = raster_arr_2
    g = raster_arr_1
    b = raster_arr_2 / raster_arr_1

    show(np.array([r,g,b]))

def show_single_band_img(path, band=1, pct=[0,100], hist=False, meta=False):

    raster = rasterio.open(path)
    if meta:
        print(raster.meta)
    if hist:
        show_hist(source=raster, bins=50, title='Histogram',
                histtype='stepfilled', alpha=0.5)
    
    raster_arr = pct_clip(raster.read(band), pct=pct)

    show(raster_arr)