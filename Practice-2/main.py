from pathlib import Path
from astropy.io import fits
import numpy as np
import astroalign as aa

from scipy.spatial.transform import Rotation
# Путь к данным до папки 
from config import data_folder
folder = Path(data_folder).expanduser()

from functions import *

def band_reader(name: str):
    band_list = []
    for file in fits_list(folder/name):
        with fits.open(file) as hdul:
            header = hdul[0].header
            data = crop(hdul[0].data, True)
            band_list.append(data)
    return band_list

bands = ['Br', 'h', 'j', 'k']
band_list = []
for band in bands:
    band_list.append(band_reader(band))

def coord_shifts(array: np.ndarray, coord: np.ndarray):
    """ Функция поиска координат звезды на разных снимках """
    radius_star = 20 
    source = array[0] #относительно этого опорного изображения будем считать сдвиги и повороты
    new_coords = np.array([coord]).reshape(1,2)
    for j in range(1, len(array)):
       target = array[j]
       transf, (source_list, target_list) = aa.find_transform(source, target) # Значения координат вшиты в параметры скобках
       for (x1, y1), (x2, y2) in zip(source_list, target_list):               # Значения угла transf.rotation в радианах
            if (x1 - coord[0])**2 + (y1 - coord[1])**2 <= radius_star:        # Значение сдвига по x,y transf.translation
                x2 = f"{x2:.2f}"                                              # Есть еще transf.scale, типа масштаб, но я не понял как он работает
                y2 = f"{y2:.2f}"
                new_coord = np.array([x2,y2])
                new_coords = np.append(new_coords, [new_coord], axis=0)
    return new_coords 


star_centers = np.array([[84, 105], [81,106], [81,82], [78,105]]) # Центры пульсара в разных фильтрах в первых fits-ах (почему-то у меня и Клеопатры они разные. Я написал свои)
new_coords_Br = [] # Итоговые значения сдвигов
new_coords_h = []
new_coords_j = []
new_coords_k = [] # не используется, так как у нас не сработал для него astroalign
for i in range(len(bands)-1):
    new_coord = coord_shifts(band_list[i], star_centers[i])
    print(bands[i])
    var = bands[i]
    match var:
        case 'Br':
            new_coords_Br = new_coord
        case 'h':
            new_coords_h = new_coord
        case 'j':
            new_coords_j = new_coord
        case 'k':
            new_coords_k = new_coord
        case _:
            print('Недопустимый фильтр')
    array_s = np.clip(band_list[i][0], 0, None) # Убирает отрицательные значения
    array_s = array_s**0.25 # Усиленная гамма-коррекция
    array2img(array_s).save(f'{folder}/band_{i}_{bands[i]}.png')

print(new_coords_Br)
print(new_coords_h)
print(new_coords_j)
print(new_coords_k)
exit()

for i in range(len(bands)-1):
    array = band_list[i]
    source = array[0] #относительно этого опорного изображения будем считать сдвиги и повороты
    source_coord = np.array([[87.21, 104.30],[68.18, 157.80],[67.41, 128.99], [168.52, 121.08], [128.54, 135.77]], dtype="float64") #Координаты 4-х ярких звезд в фильтре Br
    var1 = np.array([[87.21, 104.3], [81,106], [81,82]])
    radius_star = 20
    var1_smooth = np.array([var1[i]]).reshape(1,2)
    for j in range(1, len(array)):
       target = array[j]
       transf, (source_list, target_list) = aa.find_transform(source, target)
       for (x1, y1), (x2, y2) in zip(source_list, target_list):
            if (x1 - var1[i][0])**2 + (y1 - var1[i][1])**2 <= radius_star:
                new_coord = np.array([x2,y2])
                var1_smooth = np.append(var1_smooth, [new_coord], axis=0)
                #print((x1,y1),(x2,y2))
            
                #print("({:.2f}, {:.2f}) is source --> ({:.2f}, {:.2f}) in target".format(x1, y1, x2, y2))
    print(i, '==========================')
    print(var1_smooth)
    array_s = np.clip(source, 0, None) # Убирает отрицательные значения
    array_s = array_s**0.25 # Усиленная гамма-коррекция
    array2img(array_s).save(f'{folder}/band_{i}_{bands[i]}.png')