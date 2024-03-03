from pathlib import Path
from itertools import chain
from math import ceil, floor
from copy import deepcopy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from photutils.background import Background2D
from skimage import restoration
from image_registration import chi2_shift

def fits_list(path: Path):
    """ Создаёт итератор по всем найденным в папке файлам FITS """
    return chain.from_iterable(path.glob(f'*.{ext}') for ext in ('fts', 'fit', 'fits'))

def save_histogram(array: np.ndarray, path: str):
    """ Сохраняет гистограмму """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=100)
    ax.hist(array.flatten())
    fig.savefig(path)
    plt.close(fig)

def print_min_mean_max(array: np.ndarray):
    """ Печатает характеристики распределения значений в массиве """
    print(f'Min: {array.min():.2f};\tMean: {array.mean():.2f};\tMax: {array.max():.2f}.')

def array2img(array: np.ndarray):
    """ Нормализует массив по максимальному значению и сохраняет в PNG """
    mask = 1 - np.isnan(array).astype('int')
    array = np.nan_to_num(array)
    array = array.clip(0, None) / array.max()
    array = np.stack((array, mask), axis=-1)
    array *= 255
    return Image.fromarray(array.astype('uint8'), mode='LA')

def crop(array):
    """ Инверсия вертикальной оси и обрезка чёрных краёв """
    return array[-22:0:-1,19:]

def background_subtracted(array: np.ndarray):
    """ Вычитает фон неба """
    bkg = Background2D(array, (200, 200))
    return array - bkg.background

def float_shift(array: np.ndarray, x_shift: float, y_shift: float):
    """ Cубпиксельный циклический сдвиг """
    y_len, x_len = array.shape
    x_freq = x_shift * np.fft.fftfreq(x_len)[np.newaxis,:]
    y_freq = y_shift * np.fft.fftfreq(y_len)[:,np.newaxis]
    freq_grid = x_freq + y_freq
    kernel = np.exp(-1j*2*np.pi*freq_grid)
    return np.real(np.fft.ifftn(np.fft.fftn(array) * kernel))

def aligned_cube(cube0: np.ndarray, crop: bool = False):
    """ Выравнивание каналов спектрального куба """
    bands_num, y_len0, x_len0 = cube0.shape
    green_id = ceil(bands_num / 2) - 1 # единственное неискажённое изображение будет в середине
    shifts = [(0, 0)]
    for i in range(bands_num-1):
        shifts.append(chi2_shift(cube0[i], cube0[i+1], return_error=False, upsample_factor='auto'))
    shifts = -np.array(shifts)
    print(f'{shifts=}')
    walked = np.cumsum(shifts, axis=0)
    print(f'{walked=}')
    walked -= walked[green_id] # нормирование относительно "зелёного" изображения
    print(f'{walked=}')
    walked_min = np.min(walked, axis=0)
    print(f'{walked_min=}')
    walked_max = np.max(walked, axis=0)
    print(f'{walked_max=}')
    walked_len = walked_max - walked_min
    print(f'{walked_len=}')
    if crop:
        x_len1, y_len1 = (x_len0, y_len0) - walked_len
        x_zero, y_zero = walked_max
        x_end = ceil(x_zero + x_len1)
        y_end = ceil(y_zero + y_len1)
        x_zero = floor(x_zero)
        y_zero = floor(y_zero)
    else:
        x_len1, y_len1 = (x_len0, y_len0) + walked_len
        x_zero, y_zero = -walked_min
        x_end = ceil(x_zero + x_len0)
        y_end = ceil(y_zero + y_len0)
        x_zero = ceil(x_zero)
        y_zero = ceil(y_zero)
    x_len1 = ceil(x_len1)
    y_len1 = ceil(y_len1)
    cube1 = np.empty((bands_num, y_len1, x_len1))
    if crop:
        for i in range(bands_num):
            if i == green_id:
                cube1[i] = cube0[i, y_zero:y_end, x_zero:x_end]
            else:
                print(f'Band {i}')
                x_shift, y_shift = walked[i]
                print(f'{x_shift=} {y_shift=}')
                array = float_shift(cube0[i], x_shift, y_shift)
                cube1[i] = array[y_zero:y_end, x_zero:x_end]
    else:
        cube1.fill(np.nan)
        for i in range(bands_num):
            if i == green_id:
                cube1[i, y_zero:y_end, x_zero:x_end] = cube0[i]
            else:
                x_shift, y_shift = walked[i]
                array = float_shift(cube0[i], x_shift, y_shift)
                x_ceil, y_ceil = np.ceil(np.abs(walked[i])).astype('int')
                x_floor, y_floor = np.floor(np.abs(walked[i])).astype('int')
                if 0 not in (x_floor, y_floor):
                    match f'{int(x_shift > 0)}, {int(y_shift > 0)}':
                        case '1, 1':
                            corner = deepcopy(array[:y_floor, :x_floor]) # copy
                            array[:y_ceil, :x_ceil] = np.nan # cut
                            cube1[i, y_end:y_end+y_floor, x_end:x_end+x_floor] = corner # paste
                        case '1, 0':
                            corner = deepcopy(array[-y_floor:, :x_floor]) # copy
                            array[-y_ceil:, :x_ceil] = np.nan # cut
                            cube1[i, y_zero-y_floor:y_zero, x_end:x_end+x_floor] = corner # paste
                        case '0, 1':
                            corner = deepcopy(array[:y_floor, -x_floor:]) # copy
                            array[:y_ceil, -x_ceil:] = np.nan # cut
                            cube1[i, y_end:y_end+y_floor, x_zero-x_floor:x_zero] = corner # paste
                        case '0, 0':
                            corner = deepcopy(array[-y_floor:, -x_floor:]) # copy
                            array[-y_ceil:, -x_ceil:] = np.nan # cut
                            cube1[i, y_zero-y_floor:y_zero, x_zero-x_floor:x_zero] = corner # paste
                if x_floor != 0:
                    if x_shift > 0:
                        edge = deepcopy(array[:, :x_floor]) # copy
                        array[:, :x_ceil] = np.nan # cut
                        cube1[i, y_zero:y_end, x_end:x_end+x_floor] = edge # paste
                    else:
                        edge = deepcopy(array[:, -x_floor:]) # copy
                        array[:, -x_ceil:] = np.nan # cut
                        cube1[i, y_zero:y_end, x_zero-x_floor:x_zero] = edge # paste
                if y_floor != 0:
                    if y_shift > 0:
                        edge = deepcopy(array[:y_floor, :]) # copy
                        array[:x_ceil, :] = np.nan # cut
                        cube1[i, y_end:y_end+y_floor, x_zero:x_end] = edge # paste
                    else:
                        edge = deepcopy(array[-y_floor:, :]) # copy
                        array[-y_ceil:, :] = np.nan # cut
                        cube1[i, y_zero-y_floor:y_zero, x_zero:x_end] = edge # paste
                cube1[i, y_zero:y_end, x_zero:x_end] = array
    return cube1

def shifted(reference: np.ndarray, target: np.ndarray):
    """ Определяет сдвиг и выравнивает два изображения """
    xoff, yoff = chi2_shift(reference, target, return_error=False, upsample_factor='auto')
    return float_shift(target, -xoff, -yoff)

def gaussian_array(width: int):
    """ Формирует ядро свёртки """
    side = np.linspace(-1, 1, width)
    x, y = np.meshgrid(side, side)
    return np.exp(-4*(x*x + y*y))

def one_div_x_array(width: int):
    """ Формирует ядро свёртки """
    side = np.linspace(-1, 1, width)
    x, y = np.meshgrid(side, side)
    return np.exp(-4*np.sqrt(x*x + y*y))

def deconvolved(array: np.ndarray, kernel: np.ndarray):
    """ Деконволюция с указанным ядром свёртки """
    return restoration.unsupervised_wiener(array, kernel, clip=False)[0]