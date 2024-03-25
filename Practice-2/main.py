from pathlib import Path
from astropy.io import fits
import numpy as np

# Путь к данным до папки 
from config import data_folder
folder = Path(data_folder).expanduser()

import auxiliary as aux


def band_reader(name: str):
    band_list = []
    for file in aux.fits_list(folder/name):
        with fits.open(file) as hdul:
            band_list.append(hdul[0].data[::-1,:])
    return band_list

# Импорт изображений
bands = ('Br', 'h', 'j', 'k')
band_list = []
for band in bands:
    band_list.append(band_reader(band))

# Координаты звёзд для каждого фильтра на референсных изображениях (X, Y)
varstar1 = ((103, 126), (97, 126), (123, 127), (99, 124))
varstar2 = ((84, 179), (77, 179), (103, 180), (80, 177))
refstar1 = ((83, 150), (76, 150), (103, 151), (80, 149))
refstar2 = ((185, 143), (178, 143), (204, 143), (181, 141))

# Превью референсных изображений
for i, band in enumerate(bands):
    ref = np.clip(band_list[i][0], 0, None).astype('float') # Убирает отрицательные значения
    ref[tuple(reversed(varstar1[i]))] = np.nan # Закрашивание исследуемых звёзд альфа-каналом для проверки координат
    ref[tuple(reversed(varstar2[i]))] = np.nan
    ref[tuple(reversed(refstar1[i]))] = np.nan
    ref[tuple(reversed(refstar2[i]))] = np.nan
    ref = ref**0.25 # Усиленная гамма-коррекция
    aux.array2img(ref).save(f'{folder}/band_{i}_{band}.png')

# Вычисление координат для каждого снимка
for i, band in enumerate(bands):
    new_coords = aux.coord_shifts(band_list[i], varstar1[i])
    print(f'Для фильтра {band} звезда найдена на {new_coords.shape[0]} снимках из {len(band_list[i])}')
