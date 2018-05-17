import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing import image

from keras.applications.imagenet_utils import decode_predictions
from nasnet import NASNetLarge, NASNetMobile, preprocess_input

type = 'mobile'

size = 331 if type == 'large' else 224


url = 'https://tfhub.dev/google/imagenet'
model_name = 'nasnet_%s' % type

img_path = 'images/cheetah.jpg'
img = image.load_img(img_path, target_size=(size, size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x.astype('float32')

x_ = preprocess_input(x.copy())

with tf.device('/cpu:0'):
    print("Initializing TF-Hub model")
    inputs = tf.placeholder(tf.float32, [None, size, size, 3])
    tfhub = hub.Module("%s/%s/classification/1" % (url, model_name))
    features = tfhub(inputs, signature="image_classification", as_dict=True)
    model_tfhub = tf.nn.softmax(features['default'])

    if type == 'large':
        model = NASNetLarge(input_shape=(size, size, 3), weights=None)
        model.load_weights('weights/NASNet-large.h5')
        print("Loaded NASNet Large")
    else:
        model = NASNetMobile(input_shape=(size, size, 3), weights=None)
        model.load_weights('weights/NASNet-mobile.h5')
        print("Loaded NASNet Mobile")

    preds = model.predict(x_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x /= 255.
        preds_tfhub = sess.run(model_tfhub, feed_dict={inputs: x})

    print("Model ", decode_predictions(preds))
    print("TF Hub", decode_predictions(preds_tfhub[:, 1:]))

    np.testing.assert_allclose(preds, preds_tfhub[:, 1:], atol=1e-5)