import tensorflow as tf
import numpy as np
from generator import generator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

_logger = tf.logging._logger
_logger.setLevel(0)

batch_size = 100   # batch size
cat_dim = 10   # total categorical factor
con_dim = 2    # total continuous factor
rand_dim = 38  # total random latent dimension


target_num = tf.placeholder(dtype=tf.int32, shape=batch_size)
target_cval_1 = tf.placeholder(dtype=tf.float32, shape=batch_size)
target_cval_2 = tf.placeholder(dtype=tf.float32, shape=batch_size)

z = tf.one_hot(tf.ones(batch_size, dtype=tf.int32) * target_num, depth=cat_dim)
z = tf.concat(axis=z.get_shape().ndims-1, values=[z, tf.expand_dims(target_cval_1, -1), tf.expand_dims(target_cval_2, -1)])

z = tf.concat(axis=z.get_shape().ndims-1, values=[z, tf.random_normal((batch_size, rand_dim))])

gen = tf.squeeze(generator(z), -1)

def run_generator(num, x1, x2, fig_name='sample.png'):
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('checkpoint_dir'))
        imgs = sess.run(gen, {target_num: num, target_cval_1: x1, target_cval_2:x2})

        _, ax = plt.subplots(10,10, sharex=True, sharey=True)
        for i in range(10):
            for j in range(10):
                ax[i][j].imshow(imgs[i*10+j], 'gray')
                ax[i][j].set_axis_off()
        plt.savefig(os.path.join('result/',fig_name), dpi=600)
        print 'Sample image save to "result/{0}"'.format(fig_name)
        plt.close()

a = np.random.randint(0, cat_dim, batch_size)
print a
run_generator(a,
              np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
              fig_name='fake.png')

# classified image
run_generator(np.arange(10).repeat(10), np.linspace(0, 1, 10).repeat(10), np.expand_dims(np.linspace(0, 1, 10), axis=1).repeat(10, axis=1).T.flatten(),)
