from pathlib import Path
from astropy.io import fits
import numpy as np

from scipy.spatial.transform import Rotation
# Путь к данным до папки stud
from config import data_folder
folder = Path(data_folder).expanduser()

# Функции из отдельного файла
from functions import *


# NGC 304, NGC 7465, NGC 7625 - True
edge = True

# Чтение смещений

bias_list = []

for file in fits_list(folder):
    with fits.open(file) as hdul:
        #hdul.info()
        #print(repr(hdul[0]))
        header = hdul[0].header
        if header['IMAGETYP'] == 'bias':
            data = crop(hdul[0].data, edge)
            bias_list.append(data)
            #save_histogram(data, f'{folder}/{file.stem}.png')

bias_array = np.median(np.array(bias_list), axis=0)
#save_histogram(bias_array, f'{folder}/bias_histogram.png')
array2img(bias_array).save(folder/'bias.png')


# Чтение плоских полей

flat_field_list = []

for file in fits_list(folder):
    with fits.open(file) as hdul:
        header = hdul[0].header
        if header['IMAGETYP'] != 'bias':
            data = crop(hdul[0].data, edge) - bias_array
            flat_field_list.append(data / np.median(data))
            #save_histogram(data, f'{folder}/{file.stem}.png')

flat_field_array = np.median(np.array(flat_field_list), axis=0)
flat_field_array = np.clip(flat_field_array, 0.01, None)
#save_histogram(flat_field_array, f'{folder}/flat_field_histogram.png')
array2img(flat_field_array).save(folder/'flat_field.png')


# Чтение фотографий и запись в куб

def band_reader(name: str):
    band_list = []
    exposures = []
    for file in fits_list(folder/name):
        with fits.open(file) as hdul:
            header = hdul[0].header
            exposures.append(header['EXPTIME'])
            data = (crop(hdul[0].data, edge) - bias_array) / flat_field_array
            data = cosmic_ray_subtracted(data, sigma=5)
            data = background_subtracted(data, size_px=200)
            band_list.append(data)
            #save_histogram(data, f'{folder}/{name}/{file.stem}.png')
    cube = aligned_cube(np.array(band_list), crop=False)
    return smart_mean(cube, exposures, crop=False)

bands = ('B', 'V', 'R', 'I')
#bands = ('B', 'V', 'R') # PGC 60020 и UGC 1198
band_list = []
for band in bands:
    band_list.append(band_reader(band))

photospectral_cube = aligned_cube(band_list, crop=False)


# Сохранение нормированных на максимальную яркость фотографий

for i in range(len(bands)):
    #save_histogram(photospectral_cube[i], f'{folder}/hist_{i}_{bands[i]}.png')
    array = photospectral_cube[i]
    array = np.clip(array, 0, None) # Убирает отрицательные значения
    array = array**0.25 # Усиленная гамма-коррекция
    array2img(array).save(f'{folder}/band_{i}_{bands[i]}.png')


# Удаление фильтра I
photospectral_cube = np.nan_to_num(trimmed_nan(photospectral_cube[0:3], crop=True))

photospectral_cube[0] *= 0.064
photospectral_cube[1] *= 0.040
photospectral_cube[2] *= 0.012


# Размеры ядер свёртки для разных галактик

# NGC 304
#kernel_sizes = (9, 9, 8)

# NGC 7465
kernel_sizes = (11, 11, 9)

# NGC 7625
#kernel_sizes = (9, 8, 9)
#photospectral_cube[1] *= 0.8
#photospectral_cube[2] *= 1.5

# PGC 60020
#kernel_sizes = (17, 13, 13)
#photospectral_cube[0] *= 1.6

# UGC 1198
#kernel_sizes = (9, 9, 8)


for i in range(3):
    array = deconvolved(photospectral_cube[i], one_div_x_array(kernel_sizes[i]))
    array = np.clip(array*0.5, 0, 1)**(1/2.2) * 255
    Image.fromarray(array.astype('int8'), mode='L').save(f'{folder}/color_{i}.png')

# Сохранение среднего результата, нормированного по одному из пикселей
array = smart_mean(photospectral_cube, [1, 1, 1], crop=False)
#array = deconvolved(array, one_div_x_array(11)) # Убирает размытие
array = np.clip(array, 0, None) # Убирает отрицательные значения
result = array**0.25 # Усиленная гамма-коррекция
array2img(result).save(f'{folder}/result_1_4.png')


# ФОТОМЕТРИЯ - анализ photospectral_cube относительно центра галактики


angle_small = 26.48

from scipy import ndimage, datasets

import matplotlib.pyplot as plt


array_rot = np.array(ndimage.rotate(result, 18.4, reshape=False))
array2img(array_rot).save(f'{folder}/array_rot.png')

center_coord = np.array([520, 462])

coord_big = np.array([373+16, 694-8])
#slice_big = array_rot[coord_big[0]:coord_big[1] , center_coord[1]]
slice_big = array_rot[center_coord[1], coord_big[0]:coord_big[1]]

coord_small = np.array([330, 626])
#slice_small = array_rot[center_coord[0], coord_small[0]:coord_small[1]]
slice_small = array_rot[ coord_small[0]:coord_small[1], center_coord[0]]


import plotly.graph_objects as go
#строим графики срезов
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=np.arange(slice_big.size), y=slice_big,
#                    mode='lines',
#                    name='lines'))
#fig.add_trace(go.Scatter(x=np.arange(slice_small.size), y=slice_small,
#                    mode='lines',
#                    name='lines'))
#fig.show()


#Выполнить интегральную фотометрию галактики

center_BVRI = np.array([[540, 525],
                       [530, 511],
                       [531, 494],
                       [531, 478]])

def circular_mask(shape, center, radius):
    x, y = np.ogrid[:shape[0], :shape[1]]
    mask = (x - center[0])*2 + (y - center[1])*2 <= radius**2
    return mask

integral_sum = np.array([[0.0 for _ in range(89)] for _ in range(3)])

radius = np.arange(3, 270, 3)


for i in range(3):
    integral = np.array([])
    for rad in radius:
        mask = circular_mask(photospectral_cube[i].shape, center_BVRI[i], rad)
        result_circ = photospectral_cube[i][mask]
        integral = np.append(integral, sum(result_circ))
    #print(integral)
    integral_sum[i]= integral

#print(integral_sum)


fig = go.Figure()
fig.add_trace(go.Scatter(x=radius, y=integral_sum[0],
                    mode='lines',
                    name='B'))
fig.add_trace(go.Scatter(x=radius, y=integral_sum[1],
                    mode='lines',
                    name='V'))
fig.add_trace(go.Scatter(x=radius, y=integral_sum[2],
                    mode='lines',
                    name='R'))
fig.show()