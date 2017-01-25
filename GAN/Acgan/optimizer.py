from utils import Opt
import tensorflow as tf

def optim(loss, **kwargs):
    r"""Applies gradients to variables.

    Args:
        loss: A 0-D `Tensor` containing the value to minimize.
        kwargs:
          optim: A name for optimizer. 'MaxProp' (default), 'AdaMax', 'Adam', or 'sgd'.
          lr: A Python Scalar (optional). Learning rate. Default is .001.
          beta1: A Python Scalar (optional). Default is .9.
          beta2: A Python Scalar (optional). Default is .99.
          category: A string or string list. Specifies the variables that should be trained (optional).
            Only if the name of a trainable variable starts with `category`, it's value is updated.
            Default is '', which means all trainable variables are updated.
    """
    opt = Opt(kwargs)
    # opt += Opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='')

    # default training options
    opt += Opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='')

    # select optimizer
    # if opt.optim == 'MaxProp':
        # optim = tf.sg_optimize.MaxPropOptimizer(learning_rate=opt.lr, beta2=opt.beta2)
    # elif opt.optim == 'AdaMax':
        # optim = tf.sg_optimize.AdaMaxOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)
    # elif opt.optim == 'Adam':
    if opt.optim == 'Adm':
        optim = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)
    else:
        optim = tf.train.GradientDescentOptimizer(learning_rate=opt.lr)

    # get trainable variables
    if isinstance(opt.category, (tuple, list)):
        var_list = []
        for cat in opt.category:
            var_list.extend([t for t in tf.trainable_variables() if t.name.startswith(cat)])
    else:
        var_list = [t for t in tf.trainable_variables() if t.name.startswith(opt.category)]

    # calc gradient
    gradient = optim.compute_gradients(loss, var_list=var_list)

    # add summary
    for v, g in zip(var_list, gradient):
        # exclude batch normal statics
        if 'mean' not in v.name and 'variance' not in v.name \
                and 'beta' not in v.name and 'gamma' not in v.name:
                prefix = ''
                # summary name
                name = prefix + ''.join(v.name.split(':')[:-1])
                # summary statistics
                # noinspection PyBroadException
                try:
                    tf.summary.scalar(name + '/grad', tf.global_norm([g]))
                    tf.summary.histogram(name + '/grad-h', g)
                except:
                    pass
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # gradient update op
    return optim.apply_gradients(gradient, global_step=global_step), global_step