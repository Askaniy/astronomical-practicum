from pathlib import Path
from astropy.io import fits
import numpy as np
from ccdproc import cosmicray_lacosmic

# Путь к данным до папки stud
from config import data_folder
folder = Path(data_folder).expanduser()

# Функции из отдельного файла
from functions import *


# Чтение смещений

bias_list = []

for file in fits_list(folder):
    with fits.open(file) as hdul:
        #hdul.info()
        #print(repr(hdul[0]))
        header = hdul[0].header
        if header['IMAGETYP'] == 'bias':
            data = crop(hdul[0].data)
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
            data = crop(hdul[0].data) - bias_array
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
            data = (crop(hdul[0].data) - bias_array) / flat_field_array
            data = cosmicray_lacosmic(data, sigclip=5)[0]
            data = background_subtracted(data)
            band_list.append(data)
            #save_histogram(data, f'{folder}/{name}/{file.stem}.png')
    cube = aligned_cube(np.array(band_list), crop=False)
    return np.average(cube, axis=0, weights=exposures)

bands = ('B', 'V', 'R', 'I')
band_list = []
for band in bands:
    band_list.append(band_reader(band))

for i in range(len(bands)):
    #save_histogram(photospectral_cube[i], f'{folder}/hist_{i}_{bands[i]}.png')
    array = band_list[i]
    #array = deconvolved(array, one_div_x_array(11)) # Убирает размытие
    array = np.clip(array, 0, None) # Убирает отрицательные значения
    array = array**0.25 # Усиленная гамма-коррекция
    array2img(array).save(f'{folder}/-band_{i}_{bands[i]}.png')

photospectral_cube = aligned_cube(band_list, crop=True)


# Сохранение нормированных на максимальную яркость фотографий

for i in range(len(bands)):
    #save_histogram(photospectral_cube[i], f'{folder}/hist_{i}_{bands[i]}.png')
    array = photospectral_cube[i]
    #array = deconvolved(array, one_div_x_array(11)) # Убирает размытие
    array = np.clip(array, 0, None) # Убирает отрицательные значения
    array = array**0.25 # Усиленная гамма-коррекция
    array2img(array).save(f'{folder}/band_{i}_{bands[i]}.png')
