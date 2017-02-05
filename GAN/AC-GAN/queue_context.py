import tensorflow as tf

def queue_context(sess=None):
    r"""Context helper for queue routines.

    Args:
      sess: A session to open queues. If not specified, a new session is created.

    Returns:
      None
    """

    # default session
    sess = tf.get_default_session() if sess is None else sess

    # thread coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return coord, threads