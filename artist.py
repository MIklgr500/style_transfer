import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models import ArtistModel
from config import ArtistConfig
from losses import content_loss, style_loss
from utils import load_style_img, load_content_img, canvas_creator


class Artist:
    def __init__(self,
                 model: ArtistModel):
        self._model = model

    def transform(self, config: ArtistConfig):
        content_img = np.expand_dims(load_content_img(config), axis=0)
        style_img = np.expand_dims(load_style_img(config), axis=0)
        canvas_img = np.expand_dims(canvas_creator(config), axis=0)

        content_img = self._model.preprocess(content_img)
        style_img = self._model.preprocess(style_img)
        canvas_img = self._model.preprocess(canvas_img)

        network = self._model.build()

        with tf.Session() as sess:
            loss = self._total_loss(sess,
                                    network,
                                    style_img,
                                    content_img,
                                    config)

            self._minimize(sess,
                           network,
                           canvas_img,
                           loss,
                           self._model.postprocess,
                           config)

            output_img = sess.run(network['input'])

        return self._model.postprocess(output_img)

    @staticmethod
    def _total_style_loss(sess,
                          network,
                          style,
                          config: ArtistConfig):
        sess.run(network['input'].assign(style))
        loss = 0.
        if config.verbose:
            print('Style Layer: ')
        for indx, w in zip(config.style_layers, config.style_layer_weights):
            if config.verbose:
                print(f'\t{list(network.keys())[indx]}')
            a = tf.convert_to_tensor(sess.run(network[list(network.keys())[indx]]))
            x = network[list(network.keys())[indx]]
            loss += style_loss(a, x)
        loss /= float(len(config.style_layer_weights))
        return loss

    @staticmethod
    def _total_content_loss(sess,
                            network,
                            content,
                            config: ArtistConfig):
        sess.run(network['input'].assign(content))
        loss = 0.
        if config.verbose:
            print('Content Layer: ')
        for indx, w in zip(config.content_layers, config.content_layer_weights):
            if config.verbose:
                print(f'\t{list(network.keys())[indx]}')
            p = tf.convert_to_tensor(sess.run(network[list(network.keys())[indx]]))
            x = network[list(network.keys())[indx]]
            loss += content_loss(p, x)
        loss /= float(len(config.content_layer_weights))
        return loss

    @staticmethod
    def _total_loss(sess,
                    network,
                    style,
                    content,
                    config: ArtistConfig):
        s_loss = Artist._total_style_loss(sess, network, style, config)
        c_loss = Artist._total_content_loss(sess, network, content, config)
        v_loss = tf.image.total_variation(network['input'])

        return s_loss*config.beta + c_loss*config.alpha + config.gamma*v_loss

    @staticmethod
    def _minimize(sess,
                  network,
                  init_img,
                  loss,
                  postproc,
                  config: ArtistConfig):
        if config.optimizer == 'lbfgs':
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method='L-BFGS-B',
                options={'maxiter': config.n_iter,
                         'disp':1 if config.verbose else 0})
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(network['input'].assign(init_img))
            optimizer.minimize(sess)

        elif config.optimizer == 'adam':
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(config.lr, global_step,
                                                       50, 0.95, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            sess.run(network['input'].assign(init_img))

            for iteration in range(config.n_iter):
                sess.run(train_op)
                if config.verbose and iteration % 10 == 0:
                    curr_loss = sess.run(loss)
                    if iteration % 30 == 0 and config.debug:
                        output_img = sess.run(network['input'])
                        plt.imshow(postproc(output_img))
                        plt.show()
                    print("At iterate {}\tloss = {}".format(iteration, curr_loss[0]))
                iteration += 1
        else:
            raise ModuleNotFoundError(f'Undefined optimizer {config.optimizer}')
