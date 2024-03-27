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
            band_list.append(hdul[0].data[::-1,:].astype('float'))
    return band_list

# Импорт изображений
bands = ('Br', 'h', 'j', 'k')
band_list = []
for band in bands:
    band_list.append(band_reader(band))

# Координаты звёзд для каждого фильтра на референсных изображениях (X, Y)
varstar1 = ((100, 124), (117, 124), (116, 125), (117, 124))
varstar2 = ((80, 178), (97, 178), (96, 179), (98, 178))
refstar1 = ((80, 149), (97, 149), (96, 150), (97, 149))
refstar2 = ((181, 141), (198, 141), (197, 142), (198, 141))

# Превью референсных изображений
for i, band in enumerate(bands):
    ref = np.clip(band_list[i][0], 0, None) # Убирает отрицательные значения
    ref[tuple(reversed(varstar1[i]))] = np.nan # Закрашивание исследуемых звёзд альфа-каналом для проверки координат
    ref[tuple(reversed(varstar2[i]))] = np.nan
    ref[tuple(reversed(refstar1[i]))] = np.nan
    ref[tuple(reversed(refstar2[i]))] = np.nan
    ref = ref**0.25 # Усиленная гамма-коррекция
    aux.array2img(aux.scale_array(ref, 4)).save(f'{folder}/band_{i}_{band}.png')


# Эксперименты над PSF

data = band_list[0][0]

from photutils.psf import IntegratedGaussianPRF, PSFPhotometry, make_psf_model

# Стандартная гауссиана
psf_model1 = IntegratedGaussianPRF(flux=1, sigma=2.7/2.35)

# Кастомная модель в оболочке astropy
psf_model4 = make_psf_model(aux.custom_psf1())
psf_model5 = make_psf_model(aux.custom_psf2())

# Вписывание в координаты
psfphot = PSFPhotometry(psf_model5, fit_shape=(15, 15), aperture_radius=7)
phot = psfphot(data, init_params=aux.coords2table(varstar2[0]))
print(phot)

# Просмотр результата
resid = psfphot.make_residual_image(data, (15, 15))
aux.array2img(aux.scale_array(np.clip(resid, 0, None)**0.25, 4)).save(f'{folder}/band_1_Br_.png')

exit()

# Вычисление координат для каждого снимка
for i, band in enumerate(bands):
    new_coords = aux.coord_shifts(band_list[i], varstar1[i])
    print(f'Для фильтра {band} звезда найдена на {new_coords.shape[0]} снимках из {len(band_list[i])}')
