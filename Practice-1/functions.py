from pathlib import Path
from itertools import chain
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from photutils.background import Background2D
from skimage import restoration
from image_registration import chi2_shift
from image_registration.fft_tools import shift

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
    array_norm = array / array.max() * 255
    img = Image.fromarray(array_norm.clip(0, 255).astype('uint8'), mode='L')
    return img

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

def shifted(reference: np.ndarray, target: np.ndarray):
    """ Определяет сдвиг и выравнивает два изображения """
    xoff, yoff = chi2_shift(reference, target, return_error=False, upsample_factor='auto')
    return float_shift(target, -xoff, -yoff)

def gaussian_array(width: int):
    """ Формирует ядро свёртки """
    side = np.linspace(-1, 1, width)
    x, y = np.meshgrid(side, side)
    return np.exp(-4*(x*x + y*y))

def one_dix_x_array(width: int):
    """ Формирует ядро свёртки """
    side = np.linspace(-1, 1, width)
    x, y = np.meshgrid(side, side)
    return np.exp(-4*np.sqrt(x*x + y*y))

def deconvolved(array: np.ndarray, kernel: np.ndarray):
    """ Деконволюция с указанным ядром свёртки """
    return restoration.unsupervised_wiener(array, kernel, clip=False)[0]