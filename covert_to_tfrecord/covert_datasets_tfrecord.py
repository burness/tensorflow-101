#-*-coding:utf-8-*-

"""
This script is used to convert the images dataset of folder to tfrecord.
The image data set is expected to reside in JPEG files located in the following directory structure.

    data_dir/label_0/image0.jpg
    data_dir/label_1/image1.jpg
    ...
And this script will converts the traning and eval data into a sharded data set consisting of TFRecord files

    train_dir/train-00000-of-01024
    train_dir/train-00001-of-01024
    ...
and
    val_dir/validation-00000-of-00128
    val_dir/validation-00001-0f-00128
    ...
"""
import tensorflow as tf
import os
import random
import math
import sys

_NUM_SHARDS = 2
_RANDOM_SEED = 0
_NUM_VALIDATION = 1000
LABELS_FILENAME = 'labels.txt'


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image



def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = "102flowers_%s_%05d-of-%05d.tfrecord"%(
        split_name, shard_id, _NUM_SHARDS
    )
    return os.path.join(dataset_dir, output_filename)

def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id
            )
            if not tf.gfile.Exists(output_filename):
                return False
    return True

def _get_filenames_and_classes(dataset_dir, dataset_name):
    sex_detection_root = os.path.join(dataset_dir, dataset_name)
    direcotries = []
    class_names = []
    for filename in os.listdir(sex_detection_root):
        path = os.path.join(sex_detection_root, filename)
        if os.path.isdir(path):
            direcotries.append(path)
            class_names.append(filename)
    
    image_filenames = []
    for dir in direcotries:
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            image_filenames.append(path)
    return image_filenames, sorted(class_names)

def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

# def _clean_up_temporary_files(dataset_dir, dataset_name):
#   """Removes temporary files used to create the dataset.

#   Args:
#     dataset_dir: The directory where the temporary files are stored.
#   """
#   filename = _DATA_URL.split('/')[-1]
#   filepath = os.path.join(dataset_dir, filename)
#   tf.gfile.Remove(filepath)

#   tmp_dir = os.path.join(dataset_dir, dataset_name)
#   tf.gfile.DeleteRecursively(tmp_dir)

def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'validation']
    num_per_shard = int(math.ceil(len(filenames)/float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id
                )
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1)*num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        # print(filenames[i+1])
                        sys.stdout.write('\r>> Converting image %d/%d shard %d'%(
                            i+1, len(filenames), shard_id
                        ))
                        sys.stdout.flush()
                        image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = image_to_tfexample(image_data, 'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def run(dataset_dir, dataset_name):
    """
    Args: dataset_dir : the dataset dir where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    
    if _dataset_exists(dataset_dir):
        print("Dataset files already exist. Existing without re-creating them.")
    
    images_filenames, class_names = _get_filenames_and_classes(dataset_dir,dataset_name)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    random.seed(_RANDOM_SEED)
    random.shuffle(images_filenames)
    training_filenames = images_filenames[_NUM_VALIDATION:]
    validation_filenames = images_filenames[:_NUM_VALIDATION]
    _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir)

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, dataset_dir)

    # _clean_up_temporary_files(dataset_dir,dataset_name)
    print('\n Finised convering the sex detection dataset!')






