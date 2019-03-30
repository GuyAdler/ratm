import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    Fy = gaussian_mask(ap[..., 0::2], glimpse_size[0], inpt_H)
    Fx = gaussian_mask(ap[..., 1::2], glimpse_size[1], inpt_W)

    gs = []
    for channel in tf.unstack(inpt, axis=rank - 1):
        g = tf.matmul(tf.matmul(Fy, channel, adjoint_a=True), Fx)
        gs.append(g)
    g = tf.stack(gs, axis=rank - 1)

    return g


class RATMAttention(object):
    """Implemented after https://arxiv.org/abs/1510.08660"""
    n_params = 6

    def __init__(self, input_size, glimpse_size):
        self.input_size = np.asarray(input_size, dtype=np.int32)
        self.glimpse_size = np.asarray(glimpse_size, dtype=np.int32)

    def extract_glimpse(self, inpt, raw_att):
        raw_att_flat = tf.reshape(raw_att, (-1, self.n_params), 'flat_raw_att')
        glimpse = extract_glimpse(inpt, raw_att_flat, self.glimpse_size)
        return glimpse

    def attention_to_bbox(self, att):
        with tf.variable_scope('attention_to_bbox'):
            yx = att[..., :2]
            hw = att[..., 2:4] * (self.glimpse_size[tf.newaxis, :2] - 1)
            bbox = tf.concat([yx, hw], axis=1)
            # bbox.set_shape(att.get_shape()[:-1].concatenate((4,)))
        return bbox

    # def attention_region(self, att):
    #     return self.attention_to_bbox(att)

    def bbox_to_attention(self, bbox):
        with tf.variable_scope('ratm_bbox_to_attention'):
            us = bbox[:, :2]
            ss = tf.divide(bbox[..., 2:], self.glimpse_size[tf.newaxis, :2])
            ds = tf.divide(bbox[..., 2:], (self.glimpse_size[tf.newaxis, :2] - 1.))

            att = tf.concat([us, ss, ds], axis=1)
        return att

    # @staticmethod
    # def _to_axis_attention(params, glimpse_dim, inpt_dim):
    #     u, s, d = (params[..., i] for i in range(RATMAttention.n_params // 2))
    #     u = u * inpt_dim
    #     s = (s + 1e-5) * float(inpt_dim) / glimpse_dim
    #     d = d * float(inpt_dim - 1) / (glimpse_dim - 1)
    #     return u, s, d

    # def _to_attention(self, params):
    #     (y, x), (u, v) = self.input_size[:2], self.glimpse_size[:2]
    #     uy, sy, dy = self._to_axis_attention(params[..., ::2], u, y)
    #     ux, sx, dx = self._to_axis_attention(params[..., 1::2], v, x)
    #
    #     ap = (uy, ux, sy, sx, dy, dx)
    #     ap = tf.transpose(tf.stack(ap), name='attention')
    #     assert ap.get_shape()[-1] == self.n_params, 'Invalid attention shape={}!'.format(ap.get_shape())
    #     return ap


class AttentionCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, glimpse_size, feature_extractor, share_branch_weights=True):
        self.glimpse_size = glimpse_size
        self.feature_extractor = feature_extractor
        self.share_weights = share_branch_weights
        self.input_size = input_size
        self.ratm_unit = RATMAttention(self.input_size, self.glimpse_size)

    def __call__(self, input, state, scope=None):
        attention_params = self.ratm_unit.bbox_to_attention(state)
        glimpse_tensor = self.ratm_unit.extract_glimpse(input, attention_params)
        deep_features = self.feature_extractor(glimpse_tensor)
        new_state = self.ratm_unit.attention_to_bbox(attention_params)

        flat_output = tf.layers.flatten(deep_features)

        return flat_output, new_state

    @property
    def state_size(self):
        return self.ratm_unit.n_params

    @property
    def output_size(self):
        output_size = self.glimpse_size[0]*self.glimpse_size[1]*3
        return output_size

    def zero_state(self, batch_size, bbox0, dtype):
        initial_state = bbox0

        return initial_state


cat_filename = "cat.jpg"
cat = cv2.imread(cat_filename).astype('float')[:, :, ::-1]/255
init_bbox = np.array([[360, 870, 112, 112]], dtype=np.float32)
# y, x, h, w
cat_shape = cat.shape

identity = tf.identity
MyRNNCell = AttentionCell(cat_shape[:2], (224, 224), identity)
input_pl = tf.placeholder(tf.float32, shape=[1, 5] + list(cat_shape))

initial_state = MyRNNCell.zero_state(1, init_bbox, dtype=tf.float32)

output, final_state = tf.nn.dynamic_rnn(MyRNNCell, input_pl, initial_state=initial_state, dtype=tf.float32, time_major=False)

cat_catenate = np.tile(cat[np.newaxis, np.newaxis, :, :, :], (1, 5, 1, 1, 1))

with tf.Session() as sess:
    out, state = sess.run([output, final_state], feed_dict={input_pl: cat_catenate})

out_images = np.reshape(out, [1, 5, 224, 224, 3])
plt.imshow(out_images[0, 0])
plt.show()
