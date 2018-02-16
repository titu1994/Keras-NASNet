from __future__ import print_function
from __future__ import absolute_import

from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt

def cutout(img):
    MAX_CUTS = 5  # chance to get more cuts
    MAX_LENGTH_MULTIPLIER = 5  # change to get larger cuts ; 16 for cifar 10, 8 for cifar 100

    height = img.shape[1]
    width = img.shape[2]

    # normalize before adding the mask
    mean = img.mean(keepdims=True)
    img -= mean

    mask = np.ones((height, width), np.float32)
    nb_cuts = np.random.randint(1, MAX_CUTS + 1)

    for i in range(nb_cuts):
        y = np.random.randint(height)
        x = np.random.randint(width)
        length = 4 * np.random.randint(1, MAX_LENGTH_MULTIPLIER + 1)

        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)

        mask[y1: y2, x1: x2] = 0.

    # apply mask
    img = img * mask

    # denormalize
    img += mean

    return img


if __name__ == '__main__':
    size = 256

    img_path = 'images/cheetah.jpg'
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    x = (x / 255.0).astype('float32')

    x = cutout(x)

    plt.imshow(x)
    plt.show()
