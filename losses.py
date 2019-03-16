import tensorflow as tf


def content_loss(p, x):
    loss = 1./float(2) * tf.reduce_sum(tf.pow(x - p, 2))
    return loss


def gram_matrix(x, area, channel):
    F = tf.reshape(x, (area, channel))
    G = tf.matmul(F, F, adjoint_a=True)
    return G


def style_loss(a, x):
    _, h, w, c = a.shape

    M = h.value*w.value
    N = c.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = 1./float(4 * N**2 * M**2) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss
