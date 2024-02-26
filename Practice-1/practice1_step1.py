from pathlib import Path
from itertools import chain
from astropy.io import fits
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from config import data_folder
folder = Path(data_folder).expanduser()


def fits_list(path: Path):
    return chain.from_iterable(path.glob(f'*.{ext}') for ext in ('fts', 'fit', 'fits'))

def save_histogram(array: np.ndarray, path: str):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
    ax.hist(array.flatten(), bins='auto')
    fig.savefig(path)
    plt.close(fig)

def print_min_mean_max(array: np.ndarray):
    print(f'Min: {array.min():.2f};\tMean: {array.mean():.2f};\tMax: {array.max():.2f}.')

def array2img(array: np.ndarray):
    array_norm = array / array.max() * 255
    img = Image.fromarray(array_norm.clip(0, 255).astype('uint8'), mode='L')
    return img


# Чтение смещений

bias_list = []

for file in fits_list(folder):
    with fits.open(file) as hdul:
        #hdul.info()
        #print(repr(hdul[0]))
        header = hdul[0].header
        if header['IMAGETYP'] == 'bias':
            data = hdul[0].data
            bias_list.append(data)
            #save_histogram(data, f'{folder}/{file.stem}.png')

bias_array = np.median(np.array(bias_list), axis=0)
save_histogram(bias_array, f'{folder}/bias_histogram.png')
array2img(bias_array).save(folder/'bias.png')


# Чтение плоских полей

flat_field_list = []

for file in fits_list(folder):
    with fits.open(file) as hdul:
        header = hdul[0].header
        if header['IMAGETYP'] != 'bias':
            data = (hdul[0].data - bias_array) / header['EXPTIME']
            flat_field_list.append(data)
            #save_histogram(data, f'{folder}/{file.stem}.png')

flat_field_array = np.median(np.array(flat_field_list), axis=0)
flat_field_array /= flat_field_array.mean()
flat_field_array = np.clip(flat_field_array, 0.01, None)
save_histogram(flat_field_array, f'{folder}/flat_field_histogram.png')
array2img(flat_field_array).save(folder/'flat_field.png')


def band_reader(name: str):
    band_list = []
    for file in fits_list(folder/name):
        with fits.open(file) as hdul:
            header = hdul[0].header
            data = np.clip(hdul[0].data - bias_array, 0, None) / header['EXPTIME']
            band_list.append(data)
            #save_histogram(data, f'{folder}/{name}/{file.stem}.png')
    return np.mean(np.array(band_list), axis=0)

bands = ('B', 'V', 'R', 'I')
band_list = []
for band in bands:
    band_list.append(band_reader(band))

photospectral_cube = np.array(band_list) / flat_field_array
photospectral_cube = photospectral_cube[:,::-1,:] # инверсия вертикальной оси
photospectral_cube = photospectral_cube[:,20:,19:] # обрезка чёрных краёв

for i in range(len(bands)):
    img = array2img(photospectral_cube[i])
    img.save(f'{folder}/band_{i}_{bands[i]}.png')
