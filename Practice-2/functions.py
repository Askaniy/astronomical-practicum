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

def coord_shifts(array: np.ndarray, coord: np.ndarray):
    """ Функция поиска координат звезды на разных снимках """
    radius_star = 20 
    reference = array[0] # относительно этого опорного изображения будем считать сдвиги и повороты
    new_coords = [coord]
    for j in range(1, len(array)):
       target = array[j]
       transf, (reference_list, target_list) = aa.find_transform(reference, target) # Значения координат вшиты в параметры скобках
       for (x1, y1), (x2, y2) in zip(reference_list, target_list):      # Значения угла transf.rotation в радианах
            if (x1 - coord[0])**2 + (y1 - coord[1])**2 <= radius_star:  # Значение сдвига по x,y transf.translation, есть еще transf.scale, типа масштаб, но я не понял как он работает
                new_coords.append([x2, y2])
    return np.array(new_coords)