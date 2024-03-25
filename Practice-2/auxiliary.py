from pathlib import Path
from itertools import chain
from PIL import Image
import numpy as np

def fits_list(path: Path):
    """ Создаёт итератор по всем найденным в папке файлам FITS """
    return chain.from_iterable(path.glob(f'*.{ext}') for ext in ('fts', 'fit', 'fits', 'FTS', 'FIT', 'FITS'))

def print_min_mean_max(array: np.ndarray):
    """ Печатает характеристики распределения значений в массиве """
    print(f'Min: {array.min():.2f};\tMean: {array.mean():.2f};\tMax: {array.max():.2f}.')

def scale_array(arr: np.ndarray, times: int):
    """ Масштабирование массива в целое число раз без интерполяции """
    return np.repeat(np.repeat(arr, times, axis=0), times, axis=1) 

def array2img(array: np.ndarray):
    """ Нормализует массив по максимальному значению и сохраняет в PNG """
    mask = 1 - np.isnan(array).astype('int')
    array = np.nan_to_num(array)
    array = array.clip(0, None) / array.max()
    array = np.stack((array, mask), axis=-1)
    array *= 255
    return Image.fromarray(array.astype('uint8'), mode='LA')


# Выравнивание

import astroalign as aa

def coord_shifts(array: np.ndarray, coord: tuple):
    """ Функция поиска координат звезды на разных снимках """
    radius_star = 20 # px
    reference = array[0] # опорное изображение
    new_coords = [coord]
    for target in array[1:]:
        transf, (reference_list, target_list) = aa.find_transform(reference, target)
        # Доступны transf.rotation, transf.translation, transf.scale
        for (x1, y1), (x2, y2) in zip(reference_list, target_list):
            if (x1 - coord[0])**2 + (y1 - coord[1])**2 <= radius_star:
                new_coords.append([x2, y2])
                break
    return np.array(new_coords)


# Функции PSF фотометрии

from astropy.table import QTable
from astropy.modeling import custom_model

def coords2table(coords: tuple):
    table = QTable()
    table['x'] = [coords[0]]
    table['y'] = [coords[1]]
    return table

@custom_model
def custom_psf(x, y, amplitude=1., x_0=0., y_0=0., a=0.5, b=0.15, k=0.4, p=12):
    """
    Эмпирическое ядро свёртки
    a — узость гауссианы
    b — узость центрального пика
    k — пропорция вклада гауссианы
    p — показатель степени центрального пика
    """
    # Ограничения
    k = np.clip(k, 0., 1.)
    a = np.abs(a)
    b = np.abs(b)
    p = np.clip(p, 0., None)
    # Вычисление
    r2 = (x-x_0)**2 + (y-y_0)**2
    norm = k*np.exp(-np.abs(a)*r2) + (1-k)*(1 - np.clip(b*np.sqrt(r2), 0., 1.))**p
    return amplitude * norm