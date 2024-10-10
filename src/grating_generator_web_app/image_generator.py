from PIL import Image, ImageDraw
import numpy as np

def generate_image(value):
    # Crea una imagen básica (200x200) de un color que varía según el valor recibido
    img = Image.new('RGB', (200, 200), color=(value, 100, 255 - value))
    draw = ImageDraw.Draw(img)
    
    # Dibuja un texto en la imagen para visualizar el valor
    draw.text((50, 90), f'Value: {value}', fill=(255, 255, 255))

    return img


def generate_gabor(sf_value):
    c1 = (0.75, 0.25, 0.5)
    c2 = (0.5, 0.70, 0)
    img = generate_gabor_patch_image(sf_value, 300, c1, c2)
    print(f'Gabor patch generated with spatial frequency: {sf_value}')
    draw = ImageDraw.Draw(img)
    
    # Dibuja un texto en la imagen para visualizar el valor
    draw.text((50, 90), f'Value: {sf_value}', fill=(255, 255, 255))

    return img


def generate_gabor_patch_image(frequency, size, c1, c2):
    """
    Saves a gabor patch .PNG file with the given properties
    :param frequency: The frequency of the Gabor patch
    :param size: The size in pixels of the image
    :param c1: The first color of the Gabor patch
    :param c2: The second color of the Gabor patch
    :return:
    """
    amp, f = generate_gabor_patch(frequency, size)
    
    # Convertir colores a numpy arrays y expandir dimensiones para el canal de transparencia
    c1 = np.array(c1)
    c2 = np.array(c2)
    
    im_rgb_vals = (c1 * amp[:, :, None]) + (c2 * (1 - amp[:, :, None]))
    
    alpha_channel = f
    
    im_rgba_vals = np.dstack((im_rgb_vals, alpha_channel))
    
    im = Image.fromarray((im_rgba_vals * 255).astype('uint8'), 'RGBA')
    #im.save(f"gabor_patches/custom_stim.png")

    return im


def generate_gabor_patch(frequency, size):
    im_range = np.arange(size)
    x, y = np.meshgrid(im_range, im_range)
    dx = x - size // 2
    dy = y - size // 2
    t = np.arctan2(dy, dx)
    r = np.sqrt(dx ** 2 + dy ** 2)
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    # Transición brusca para los colores (líneas) en el patrón Gabor
    amp = np.where(np.cos(2 * np.pi * (x * frequency)) >= 0, 1, 0)
    f = np.where(r <= size // 2, 1, 0)
    
    return amp, f