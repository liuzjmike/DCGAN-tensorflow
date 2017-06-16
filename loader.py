import os

import tensorflow as tf

def load(sess, var_dict, checkpoint_dir):
    saver = tf.train.Saver(var_dict)
    print(" [*] Reading checkpoints...")
    try:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    except:
        return _fail_load()
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(
            sess, os.path.join(
                checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        return fail_load()

def _fail_load():
    print(" [*] Failed to find a checkpoint")
    return False
