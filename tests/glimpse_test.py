import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def np_gaussian_mask(params, R, C):
    u, s, d = (params[..., i] for i in range(3))


def gaussian_mask(params, R, C):
    """Define a mask of size RxC given by one 1-D Gaussian per row.

    u, s and d must be 1-dimensional vectors"""
    u, s, d = (params[..., i] for i in range(3))

    for i in (u, s, d):
        assert len(u.get_shape()) == 1, i

    batch_size = tf.to_int32(tf.shape(u)[0])

    R = tf.range(tf.to_int32(R))
    C = tf.range(tf.to_int32(C))
    R = tf.to_float(R)[tf.newaxis, tf.newaxis, :]
    C = tf.to_float(C)[tf.newaxis, :, tf.newaxis]
    C = tf.tile(C, (batch_size, 1, 1))

    u, d = u[:, tf.newaxis, tf.newaxis], d[:, tf.newaxis, tf.newaxis]
    s = s[:, tf.newaxis, tf.newaxis]

    ur = u + (R - 0.) * d
    sr = tf.ones_like(ur) * s

    mask = C - ur
    mask = tf.exp(-.5 * (mask / sr) ** 2)

    mask /= tf.reduce_sum(mask, 1, keepdims=True) + 1e-8
    return mask


def extract_glimpse(inpt, attention_params, glimpse_size):
    """Extracts an attention glimpse

    :param inpt: tensor of shape == (batch_size, img_height, img_width)
    :param attention_params: tensor of shape = (batch_size, 6) as
        [uy, sy, dy, ux, sx, dx] with u - mean, s - std, d - stride"
    :param glimpse_size: 2-tuple of ints as (height, width),
        size of the extracted glimpse
    :return: tensor
    """

    ap = attention_params
    shape = inpt.get_shape()
    rank = len(shape)

    assert rank in (3, 4), "Input must be 3 or 4 dimensional tensor"

    inpt_H, inpt_W = shape[1:3]
    if rank == 3:
        inpt = inpt[..., tf.newaxis]
        rank += 1

    Fy = gaussian_mask(ap[..., 0:3], glimpse_size[0], inpt_H)
    Fx = gaussian_mask(ap[..., 3:], glimpse_size[1], inpt_W)

    # return Fy

    gs = []
    for channel in tf.unstack(inpt, axis=rank - 1):
        g = tf.matmul(tf.matmul(Fy, channel, adjoint_a=True), Fx)
        # g = tf.matmul(channel, Fx)
        gs.append(g)
    g = tf.stack(gs, axis=rank - 1)

    # g.set_shape([shape[0]] + list(glimpse_size))
    return g


cat_filename = "cat.jpg"
cat = cv2.imread(cat_filename).astype('float')[:, :, ::-1]/255
# params = np.array([[0.0, 0.5, 5.5, 0.0, 1.0, 5.5]])
params = np.array([[360.0, 0.5, (112.0/224.0), 870.0, 1.0, (112/224.0)]])
pic_placeholder = tf.placeholder(tf.float32, [None, 1200, 1600, 3])
params_placeholder = tf.placeholder(tf.float32, [None, 6])

with tf.Session() as sess:
    glimpse_op = extract_glimpse(pic_placeholder, params_placeholder, (224, 224))
    glimpse = sess.run(glimpse_op, feed_dict={pic_placeholder: np.expand_dims(cat, axis=0),
                                              params_placeholder: params})

plt.imshow(glimpse[0])
plt.show()


