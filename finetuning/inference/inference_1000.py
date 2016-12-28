import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
import sys
sys.path.append("..")
from nets.inception_v3 import *
import numpy as np
import os
import time


def get_label(sysnet_file, metadata_file):
    index_sysnet = []
    with open(sysnet_file, 'r') as fread:
        for line in fread.readlines():
            line = line.strip('\n')
            index_sysnet.append(line)
    sys_label = {}
    with open(metadata_file, 'r') as fread:
        for line in fread.readlines():
            index = line.strip('\n').split('\t')[0]
            val = line.strip('\n').split('\t')[1]
            sys_label[index] = val

    index_label = [sys_label[i] for i in index_sysnet]
    index_label.append("i don't know")
    return index_label


def inference(image):
    checkpoint_dir = '../pretrain_model'
    checkpoint_file = '../pretrain_model/inception_v3.ckpt'
    input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
    sess = tf.Session()
    arg_scope = inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v3(
            input_tensor, is_training=False, num_classes=1001)
        saver = tf.train.Saver()
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, checkpoint_file)
    im = Image.open(image).resize((299, 299))
    im = np.array(im) / 255.0
    im = im.reshape(-1, 299, 299, 3)
    start = time.time()
    predict_values, logit_values = sess.run(
        [end_points['Predictions'], logits], feed_dict={input_tensor: im})
    print 'a image take time {0}'.format(time.time() - start)
    return image, predict_values


if __name__ == "__main__":
    sample_images = './cat.jpeg'
    image, predict = inference(sample_images)

    print np.argmax(predict[0])
    index_label = get_label('./sysnet.txt', 'imagenet_metadata.txt')
    print 'the image {0}, predict label is {1}'.format(
        sample_images, index_label[np.argmax(predict[0] - 1)])
