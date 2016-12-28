import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
import sys
sys.path.append("..")
from nets.inception_v3 import *
import numpy as np
import os
import time


def inference(image):
    checkpoint_dir = '../train_log'
    input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
    sess = tf.Session()
    arg_scope = inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v3(
            input_tensor, is_training=False, num_classes=8)
        saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    im = Image.open(image).resize((299, 299))
    im = np.array(im) / 255.0
    im = im.reshape(-1, 299, 299, 3)
    start = time.time()
    predict_values, logit_values = sess.run(
        [end_points['Predictions'], logits], feed_dict={input_tensor: im})
    print 'a image take time {0}'.format(time.time() - start)
    return image, predict_values


if __name__ == "__main__":
    sample_images = 'train/ALB/img_00003.jpg'
    image, predict = inference(sample_images)
    print 'the porn score with the {0} is {1} '.format(image, predict)
