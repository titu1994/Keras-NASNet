import glob
from keras.preprocessing import image
from keras.metrics import top_k_categorical_accuracy
from keras import backend as K

import tensorflow as tf
keras_sess = tf.Session()
K.set_session(keras_sess)

from nasnet import NASNetLarge, NASNetMobile, preprocess_input, decode_predictions

import numpy as np

batch_size = 25

def process_images(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, size, size)
    image_resized = tf.cast(image_resized, tf.float32)
    image_normalized = preprocess_input(image_resized)

    label = tf.one_hot(label, depth=1000, dtype=tf.float32)
    return image_normalized, label


def generator(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(process_images, 2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)
    iterator = dataset.make_one_shot_iterator()

    batch = iterator.get_next()

    while True:
        yield K.get_session().run(batch)


type = 'mobile'

if __name__ == '__main__':
    size = 331 if type == 'large' else 224

    if tf.device('/cpu:0'):
        if type == 'large':
            model = NASNetLarge(input_shape=(size, size, 3), weights=None)
            model.load_weights('weights/NASNet-large.h5')
            print("Loaded NASNet Large")
        else:
            model = NASNetMobile(input_shape=(size, size, 3), weights=None)
            model.load_weights('weights/NASNet-mobile.h5')
            print("Loaded NASNet Mobile")
        #model.summary()


        VAL_PATH = 'D:/Yue/Documents/Downloads/ILSVRC2012_img_val/*'
        files = sorted(glob.glob(VAL_PATH))

        with open('ImageNet_Val.txt', 'r') as f:
            labels = [int(x.split()[1]) for x in f]

        files = files[:1000]
        labels = labels[:1000]

        files = np.array(files)
        labels = np.array(labels)

        model.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy', top_k_categorical_accuracy])

        scores = model.evaluate_generator(generator(batch_size), steps=len(files) // batch_size, workers=0, verbose=1)
        print("Imagenet Validation scores : ", scores)
