from pathlib import Path
from itertools import chain
from astropy.io import fits
from image_registration import chi2_shift
from image_registration.fft_tools import shift
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

def crop(array):
    """ Инверсия вертикальной оси и обрезка чёрных краёв """
    return array[-22:0:-1,19:]

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


# Чтение фотографий

def shifted(reference, target):
    xoff, yoff = chi2_shift(reference, target, return_error=False, upsample_factor='auto')
    return shift.shift2d(target, -xoff, -yoff)

def band_reader(name: str):
    band_list = []
    for file in fits_list(folder/name):
        with fits.open(file) as hdul:
            header = hdul[0].header
            data = (crop(hdul[0].data) - bias_array) / header['EXPTIME'] / flat_field_array
            band_list.append(data)
            #save_histogram(data, f'{folder}/{name}/{file.stem}.png')
    band_list[1] = shifted(band_list[0], band_list[1])
    return np.mean(np.array(band_list), axis=0)

from photutils.background import Background2D

def background_subtracted(array: np.ndarray):
    bkg = Background2D(array, (50, 50))
    return array - bkg.background

bands = ('B', 'V', 'R', 'I')
band_list = [background_subtracted(band_reader(bands[0]))]
for band in bands[1:]:
    band_list.append(shifted(band_list[0], background_subtracted(band_reader(band))))

gamma_correction = np.vectorize(lambda br: br * 12.92 if br < 0.0031308 else 1.055 * br**(1.0/2.4) - 0.055)

photospectral_cube = np.clip(np.array(band_list), 0, None)
photospectral_cube = gamma_correction(photospectral_cube / photospectral_cube.max())

for i in range(len(bands)):
    array2img(photospectral_cube[i]).save(f'{folder}/band_{i}_{bands[i]}.png')
