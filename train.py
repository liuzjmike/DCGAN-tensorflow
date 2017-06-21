import os
import numpy as np
import tensorflow as tf

from utils import pp, to_json, show_all_variables
from trainer import Trainer
from membership_trainer import MembershipTrainer

flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Epoch to train [20]")
flags.DEFINE_float(
    "learning_rate",
    0.0002,
    "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", None, "The size of train images [None]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer(
    "output_height",
    64,
    "The size of the output images to produce [64]")
flags.DEFINE_integer(
    "output_width",
    None,
    "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string(
    "name",
    "model",
    "Name of the model [model]")
flags.DEFINE_string(
    "train_set",
    None,
    "The directory that contains the images used for training. Should locate" +
    "/data. If None, same value as name [None]")
flags.DEFINE_string(
    "test_set",
    None,
    "The directory that contains the images used for testing. Should locate" +
    "/data. If None, same value as name [None]")
flags.DEFINE_string(
    "input_fname_pattern",
    "*.jpg",
    "Glob pattern of filename of input images [*.jpg]")
flags.DEFINE_string(
    "checkpoint_dir",
    "checkpoint",
    "The directory to save the checkpoints [checkpoint]")
flags.DEFINE_string(
    "sample_dir",
    "samples",
    "The directory to save the image samples [samples]")
flags.DEFINE_boolean(
    "crop",
    False,
    "Center-crop input images [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height
    if FLAGS.train_set is None:
        FLAGS.train_set = os.path.join(FLAGS.name, 'train')
    if FLAGS.test_set is None:
        FLAGS.test_set = os.path.join(FLAGS.name, 'test')

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        trainer = MembershipTrainer(
            sess,
            FLAGS.name,
            FLAGS.train_set,
            FLAGS.test_set,
            batch_size=FLAGS.batch_size,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            train_size=FLAGS.train_size,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir)

        show_all_variables()

        trainer.train(FLAGS)
        # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #                 [dcgan.h4_w, dcgan.h4_b, None])

        # Below is codes for visualization

if __name__ == '__main__':
    tf.app.run()
