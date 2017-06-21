from __future__ import division
import os
import numpy as np

from utils import heap_add_all
from trainer import Trainer, get_data

class MembershipTrainer(Trainer):
    def __init__(
            self,
            sess,
            name,
            train_set,
            test_set,
            batch_size=64,
            output_height=64,
            output_width=64,
            z_dim=100,
            gf_dim=64,
            df_dim=64,
            train_size=np.inf,
            input_fname_pattern='*.jpg',
            checkpoint_dir="checkpoint"):
        Trainer.__init__(self, sess, name, train_set, test_set, batch_size,
                         output_height, output_width, z_dim, gf_dim, df_dim,
                         train_size, input_fname_pattern, checkpoint_dir)
        self.attack_log = os.path.join(self.checkpoint_dir, 'attack_log.txt')
        test_data = get_data(test_set, input_fname_pattern)
        self.data = self.train_data + test_data
        self.in_train = [1] * len(self.train_data) + [0] * len(test_data)
        self.batch_size = batch_size

    def eval(self, epoch, config):
        Trainer.eval(self, epoch, config)
        if np.mod(epoch, 10) == 9:
            guesses = []
            for idx in xrange(len(self.data) // self.batch_size):
                in_score = self.sess.run(self.dcgan.D,
                                         feed_dict={self.dcgan.inputs:
                                                    self.get_images(self.data,
                                                                    idx, config)})
                in_label = self.in_train[idx * self.batch_size:(idx + 1) *
                                         self.batch_size]
                heap_add_all(guesses, zip(in_score, in_label),
                             max_size=len(self.train_data))
            accuracy = sum(label for score, label in guesses) / len(self.train_data)
            print('[*] Attack accuracy: %.2f' % accuracy)
            with open(self.attack_log, 'a') as f:
                f.write('Epoch [%d] Attack accuracy %.2f\n' % (epoch + 1, accuracy))
