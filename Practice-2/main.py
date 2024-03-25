from pathlib import Path
from astropy.io import fits
import numpy as np

# Путь к данным до папки 
from config import data_folder
folder = Path(data_folder).expanduser()

import functions as func


def band_reader(name: str):
    band_list = []
    for file in func.fits_list(folder/name):
        with fits.open(file) as hdul:
            band_list.append(hdul[0].data[::-1,:])
    return band_list

# Импорт изображений
bands = ('Br', 'h', 'j', 'k')
band_list = []
for band in bands:
    band_list.append(band_reader(band))

# Превью референсных изображений
for i, band in enumerate(bands):
    ref = np.clip(band_list[i][0], 0, None) # Убирает отрицательные значения
    ref = ref**0.25 # Усиленная гамма-коррекция
    func.array2img(ref).save(f'{folder}/band_{i}_{band}.png')

# Вычисление координат
star_centers = ((84, 105), (81, 106), (81, 82), (78, 105)) # Центры переменной звезды в разных фильтрах в первых fits-ах
new_coords_Br = func.coord_shifts(band_list[0], star_centers[0])
new_coords_h = func.coord_shifts(band_list[1], star_centers[1])
new_coords_j = func.coord_shifts(band_list[2], star_centers[2])
#new_coords_k = coord_shifts(band_list[3], star_centers[3]) # не используется, так как у нас не сработал для него astroalign

print(new_coords_Br)
print(new_coords_h)
print(new_coords_j)
#print(new_coords_k)
