import os
import numpy as np
import tensorflow as tf

from loader import load
from meta_saver import load_meta
from ops import deconv2d, batch_norm
from generator import conv_out_size_same

class Deconv3Visualizer(object):
    def __init__(
            self,
            sess,
            model_id,
            checkpoint_dir='checkpoint'):
        checkpoint_dir = os.path.join(checkpoint_dir, model_id)
        self.sess = sess
        meta = load_meta(checkpoint_dir)
        with tf.variable_scope("generator"):
            bn3 = batch_norm(name='g_bn3')

            s_h2 = conv_out_size_same(meta['output_height'], 2)
            s_w2 = conv_out_size_same(meta['output_width'], 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

            self.h2 = tf.placeholder(tf.float32, [meta['gf_dim'] * 2, s_h4,
                                                  s_w4, meta['gf_dim'] * 2],
                                     name='g_h2')
            h3 = deconv2d(self.h2, [meta['gf_dim'] * 2, s_h2, s_w2, meta['gf_dim']], name='g_h3')
            h3 = tf.nn.relu(bn3(h3))
            h4 = deconv2d(h3, [meta['gf_dim'] * 2, meta['output_height'], meta['output_width'], meta['channel']], name='g_h4')
            self.visualizer = tf.nn.tanh(h4)
        load(sess, tf.global_variables(), checkpoint_dir)

    def sample(self):
        h2 = np.zeros(self.h2.get_shape())
        for i in range(len(h2)):
            h2[i][2][2][i] = 1
        return self.sess.run(self.visualizer, feed_dict={self.h2: h2})

