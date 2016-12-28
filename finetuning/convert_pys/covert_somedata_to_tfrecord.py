from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import covert_datasets_tfrecord

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_integer('nFold', 1, "The nFold of Cross validation.")


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError(
            'You must supply the dataset name with --dataset_name')
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset name with --dataset_dir')

    covert_datasets_tfrecord.run(FLAGS)


if __name__ == '__main__':
    tf.app.run()