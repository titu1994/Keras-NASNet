from __future__ import print_function
from __future__ import absolute_import

from keras.preprocessing import image

import tensorflow as tf
from nasnet import NASNetLarge, NASNetMobile, preprocess_input, decode_predictions

import numpy as np

type = 'large'

if __name__ == '__main__':
    size = 331 if type == 'large' else 224

    if tf.device('/CPU:0'):
        if type == 'large':
            model = NASNetLarge(input_shape=(size, size, 3))
            print("Loaded NASNet Large")
        else:
            model = NASNetMobile(input_shape=(size, size, 3))
            print("Loaded NASNet Mobile")
        #model.summary()

        img_path = 'images/cheetah.jpg'
        img = image.load_img(img_path, target_size=(size, size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        preds = model.predict(x, verbose=1)

        print('Predicted:', decode_predictions(preds))

