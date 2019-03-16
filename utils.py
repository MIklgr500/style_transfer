import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from config import ArtistConfig


def load_content_img(config: ArtistConfig):
    img = img_to_array(load_img(config.content_path, target_size=config.size))
    return np.float32(img)


def load_style_img(config: ArtistConfig):
    img = img_to_array(load_img(config.style_path, target_size=config.size))
    return np.float32(img)


def canvas_creator(config: ArtistConfig):
    img = img_to_array(load_img(config.content_path, target_size=config.size))

    noise = np.random.uniform(127-20., 127+20., img.shape).astype(np.float32)
    return np.clip(np.float32(noise) * config.noise_rate + np.float32(img) * (1-config.noise_rate), 0, 255)
