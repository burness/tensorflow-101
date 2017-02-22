import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope

def leaky_relu(x):
     return tf.where(tf.greater(x, 0), x, 0.01 * x)

def discriminator(tensor, num_category=10, batch_size=32, num_cont=2):
    """
    """
    
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    print reuse
    print tensor.get_shape()
    with variable_scope.variable_scope('discriminator', reuse=reuse):
        tensor = slim.conv2d(tensor, num_outputs = 64, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        tensor = slim.conv2d(tensor, num_outputs=128, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        tensor = slim.flatten(tensor)
        shared_tensor = slim.fully_connected(tensor, num_outputs=1024, activation_fn = leaky_relu)
        recog_shared = slim.fully_connected(shared_tensor, num_outputs=128, activation_fn = leaky_relu)
        disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=None)
        disc = tf.squeeze(disc, -1)
        recog_cat = slim.fully_connected(recog_shared, num_outputs=num_category, activation_fn=None)
        recog_cont = slim.fully_connected(recog_shared, num_outputs=num_cont, activation_fn=tf.nn.sigmoid)
    return disc, recog_cat, recog_cont
