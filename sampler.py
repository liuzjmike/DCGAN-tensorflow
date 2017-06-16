import os
import numpy as np
import tensorflow as tf

from loader import load
from meta_saver import load_meta
from generator import Generator

class Sampler(object):
    def __init__(
            self,
            sess,
            model_id,
            checkpoint_dir='checkpoint'):
        checkpoint_dir = os.path.join(checkpoint_dir, model_id)
        meta = load_meta(checkpoint_dir)
        generator = Generator(meta['batch_size'], meta['output_height'],
                                   meta['output_width'], meta['z_dim'],
                                   meta['gf_dim'], meta['channel'])
        self.sess = sess
        self.model_id = model_id
        self.z = tf.placeholder(
            tf.float32, [generator.batch_size, generator.z_dim], name='z')
        with tf.variable_scope("generator"):
            self.sampler = generator.build_graph(self.z)
        load(sess, tf.global_variables(), checkpoint_dir)

    def sample(self, option=0):
        z = np.random.uniform(-0.5, 0.5, self.z.get_shape())
        return self.sess.run(self.sampler, feed_dict={self.z: z})
