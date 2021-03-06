import os
import numpy as np
import tensorflow as tf
import scipy.misc
from six.moves import xrange

from utils import pp
from sampler import Sampler

def get_path(fname):
    return os.path.join(FLAGS.sample_dir, FLAGS.model_id, fname)

flags = tf.app.flags
flags.DEFINE_string(
    "model_id",
    None,
    "ID of the model in the form of name_{batch size}_{output height}_{output width} [None]")
flags.DEFINE_integer("batch_size", 64, "The size of a batch [64]")
flags.DEFINE_integer(
    "batch",
    1,
    "The number of batches to produce [1]")
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
        sampler = Sampler(sess, FLAGS.model_id, FLAGS.batch_size, FLAGS.checkpoint_dir)
        for b in xrange(FLAGS.batch):
            samples = sampler.sample(option=0)
            for idx, sample in enumerate(samples):
                scipy.misc.imsave(get_path('sample_%d_%d.jpg' % (b, idx)),
                                  np.squeeze(sample))

if __name__ == '__main__':
    tf.app.run()
