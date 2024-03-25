from pathlib import Path
from itertools import chain
from PIL import Image
import numpy as np
import astroalign as aa


def fits_list(path: Path):
    """ Создаёт итератор по всем найденным в папке файлам FITS """
    return chain.from_iterable(path.glob(f'*.{ext}') for ext in ('fts', 'fit', 'fits', 'FTS', 'FIT', 'FITS'))

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

def coord_shifts(array: np.ndarray, coord: tuple):
    """ Функция поиска координат звезды на разных снимках """
    radius_star = 20 # px
    reference = array[0] # опорное изображение
    new_coords = [coord]
    for target in array[1:]:
        transf, (reference_list, target_list) = aa.find_transform(reference, target) # Значения координат вшиты в параметры скобках
        # Доступны transf.rotation, transf.translation, transf.scale
        for (x1, y1), (x2, y2) in zip(reference_list, target_list):
            if (x1 - coord[0])**2 + (y1 - coord[1])**2 <= radius_star:
                new_coords.append([x2, y2])
                break
    return np.array(new_coords)