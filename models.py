import keras
import numpy as np
import tensorflow as tf

from keras_applications.vgg19 import VGG19
from keras import backend as K

class ArtistModel:
    def __init__(self, shape, verbose):
        self._shape = shape
        self._verbose = verbose

    def build(self):
        raise NotImplementedError('Abstract')

    def preprocess(self, img):
        raise NotImplementedError('Abstract')

    def postprocess(self, img):
        raise NotImplementedError('Abstract')


class ArtistVGG19(ArtistModel):
    def __init__(self, shape,
                 verbose,
                 pooling_type='avg'):
        super().__init__(shape, verbose)
        self.pooling_type = pooling_type

    def build(self):
        if self._verbose:
            print('\n Build VGG19 architecture')
        net = {}
        h, w, c = self._shape

        if self._verbose:
            print('loading model weights...')

        weights = VGG19(include_top=False, input_shape=self._shape, weights='imagenet').get_weights()

        if self._verbose:
            print('constructing layers...')

        net['input'] = tf.Variable(np.zeros((1, h, w, c), dtype=np.float32))

        if self._verbose:
            print('LAYER GROUP 1')
        net['conv1_1'] = self._conv_layer(net['input'], W=self._get_weights(weights, 0))
        net['relu1_1'] = self._relu_layer(net['conv1_1'], b=self._get_bias(weights, 1))

        net['conv1_2'] = self._conv_layer(net['relu1_1'], W=self._get_weights(weights, 2))
        net['relu1_2'] = self._relu_layer(net['conv1_2'], b=self._get_bias(weights, 3))

        net['pool1'] = self._pool_layer(net['relu1_2'])

        if self._verbose:
            print('LAYER GROUP 2')
        net['conv2_1'] = self._conv_layer(net['pool1'], W=self._get_weights(weights, 4))
        net['relu2_1'] = self._relu_layer(net['conv2_1'], b=self._get_bias(weights, 5))

        net['conv2_2'] = self._conv_layer(net['relu2_1'], W=self._get_weights(weights, 6))
        net['relu2_2'] = self._relu_layer(net['conv2_2'], b=self._get_bias(weights, 7))

        net['pool2'] = self._pool_layer(net['relu2_2'])

        if self._verbose:
            print('LAYER GROUP 3')
        net['conv3_1'] = self._conv_layer(net['pool2'], W=self._get_weights(weights, 8))
        net['relu3_1'] = self._relu_layer(net['conv3_1'], b=self._get_bias(weights, 9))

        net['conv3_2'] = self._conv_layer(net['relu3_1'], W=self._get_weights(weights, 10))
        net['relu3_2'] = self._relu_layer(net['conv3_2'], b=self._get_bias(weights, 11))

        net['conv3_3'] = self._conv_layer(net['relu3_2'], W=self._get_weights(weights, 12))
        net['relu3_3'] = self._relu_layer(net['conv3_3'], b=self._get_bias(weights, 13))

        net['conv3_4'] = self._conv_layer(net['relu3_3'], W=self._get_weights(weights, 14))
        net['relu3_4'] = self._relu_layer(net['conv3_4'], b=self._get_bias(weights, 15))

        net['pool3'] = self._pool_layer(net['relu3_4'])

        if self._verbose:
            print('LAYER GROUP 4')
        net['conv4_1'] = self._conv_layer(net['pool3'], W=self._get_weights(weights, 16))
        net['relu4_1'] = self._relu_layer(net['conv4_1'], b=self._get_bias(weights, 17))

        net['conv4_2'] = self._conv_layer(net['relu4_1'], W=self._get_weights(weights, 18))
        net['relu4_2'] = self._relu_layer(net['conv4_2'], b=self._get_bias(weights, 19))

        net['conv4_3'] = self._conv_layer(net['relu4_2'], W=self._get_weights(weights, 20))
        net['relu4_3'] = self._relu_layer(net['conv4_3'], b=self._get_bias(weights, 21))

        net['conv4_4'] = self._conv_layer(net['relu4_3'], W=self._get_weights(weights, 22))
        net['relu4_4'] = self._relu_layer(net['conv4_4'], b=self._get_bias(weights, 23))

        net['pool4'] = self._pool_layer(net['relu4_4'])

        if self._verbose:
            print('LAYER GROUP 5')
        net['conv5_1'] = self._conv_layer(net['pool4'], W=self._get_weights(weights, 24))
        net['relu5_1'] = self._relu_layer(net['conv5_1'], b=self._get_bias(weights, 25))

        net['conv5_2'] = self._conv_layer(net['relu5_1'], W=self._get_weights(weights, 26))
        net['relu5_2'] = self._relu_layer(net['conv5_2'], b=self._get_bias(weights, 27))

        net['conv5_3'] = self._conv_layer(net['relu5_2'], W=self._get_weights(weights, 28))
        net['relu5_3'] = self._relu_layer(net['conv5_3'], b=self._get_bias(weights, 29))

        net['conv5_4'] = self._conv_layer(net['relu5_3'], W=self._get_weights(weights, 30))
        net['relu5_4'] = self._relu_layer(net['conv5_4'], b=self._get_bias(weights, 31))

        net['pool5'] = self._pool_layer(net['relu5_4'])
        return net

    def _conv_layer(self, layer_input, W):
        conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
        return conv

    def _relu_layer(self, layer_input, b):
        relu = tf.nn.relu(layer_input + b)
        return relu

    def _pool_layer(self, layer_input):
        if self.pooling_type == 'avg':
            pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
        elif self.pooling_type == 'max':
            pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
        else:
            raise ModuleNotFoundError(f'Undefined {self.pooling_type}')
        return pool

    def preprocess(self, img):
        imgpre = np.copy(img)
        imgpre = imgpre[..., ::-1]
        imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        return imgpre

    def postprocess(self, img):
        imgpost = np.copy(img)
        imgpost += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        imgpost = imgpost[0]
        imgpost = np.clip(imgpost, 0, 255).astype('uint8')
        imgpost = imgpost[..., ::-1]
        return imgpost

    @staticmethod
    def _get_weights(weights, i):
        weights = weights[i]
        return tf.constant(weights)

    @staticmethod
    def _get_bias(weights, i):
        bias = weights[i]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return b
