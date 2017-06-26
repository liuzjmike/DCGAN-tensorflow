from __future__ import division
import os
import numpy as np
import tensorflow as tf
import scipy.misc
from six.moves import xrange

from utils import pp, save_images
from conv4_visualizer import Deconv4Visualizer
from conv3_visualizer import Deconv3Visualizer

def get_path(fname):
    return os.path.join(FLAGS.sample_dir, FLAGS.model_id, fname)

flags = tf.app.flags
flags.DEFINE_string(
    "model_id",
    None,
    "ID of the model in the form of name_{batch size}_{output height}_{output width} [None]")
flags.DEFINE_integer("layer", 4, "The layer to visualize [4]")
flags.DEFINE_string(
    "checkpoint_dir",
    "checkpoint",
    "The directory to load the checkpoints [checkpoint]")
flags.DEFINE_string(
    "sample_dir",
    "samples",
    "The directory to save the image samples [samples]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.model_id is None:
        raise ValueError('Must specify model_id')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if FLAGS.layer == 3:
            visualizer = Deconv3Visualizer(sess, FLAGS.model_id, FLAGS.checkpoint_dir)
        if FLAGS.layer == 4:
            visualizer = Deconv4Visualizer(sess, FLAGS.model_id, FLAGS.checkpoint_dir)
        samples = visualizer.sample()
        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
        manifold_w = int(np.ceil(samples.shape[0]/manifold_h))
        save_images(samples, [manifold_h, manifold_w], get_path('conv%d.png' %
                                                               FLAGS.layer))

if __name__ == '__main__':
    tf.app.run()

