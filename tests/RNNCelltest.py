import tensorflow as tf
import numpy as np


class TestCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, mult_int):
        self.mult_int = mult_int

    def __call__(self, input, state, scope=None):
        if state is None:
            batch_size = tf.shape(input)[0]
            state = self.zero_state(batch_size, dtype=tf.int32)
        new_state = state * self.mult_int
        output = input + state

        return output, new_state

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def zero_state(self, batch_size, dtype):
        zero = tf.ones([batch_size, 1], dtype=tf.int32)

        return zero


input_pl = tf.placeholder(tf.int32, shape=[2, 5, 1])
MyRNNCell = TestCell(2)

output, final_state = tf.nn.dynamic_rnn(MyRNNCell, input_pl, dtype=tf.int32)

input_array = np.array([[[1], [2], [3], [4], [5]],
                        [[9], [8], [7], [6], [1]]], dtype=np.int32)

with tf.Session() as sess:
    out, state = sess.run([output, final_state], feed_dict={input_pl: input_array})

