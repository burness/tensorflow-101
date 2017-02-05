from tensorflow.examples.tutorials.mnist import input_data
from utils import data_to_tensor, Opt





class Mnist(object):
    r"""Downloads Mnist datasets and puts them in queues.
    """
    _data_dir = './asset/data/mnist'

    def __init__(self, batch_size=128, num_epochs=30, reshape=False, one_hot=False):

        # load sg_data set
        data_set = input_data.read_data_sets(Mnist._data_dir, reshape=reshape, one_hot=one_hot)

        self.batch_size = batch_size

        # save each sg_data set
        _train = data_set.train
        _valid = data_set.validation
        _test = data_set.test

        # member initialize
        self.train, self.valid, self.test = Opt(), Opt(), Opt()

        # convert to tensor queue
        self.train.image, self.train.label = \
            data_to_tensor([_train.images, _train.labels.astype('int32')], batch_size, name='train')
        self.valid.image, self.valid.label = \
            data_to_tensor([_valid.images, _valid.labels.astype('int32')], batch_size, name='valid')
        self.test.image, self.test.label = \
            data_to_tensor([_test.images, _test.labels.astype('int32')], batch_size, name='test')

        # calc total batch count
        self.train.num_batch = _train.labels.shape[0] // batch_size
        self.valid.num_batch = _valid.labels.shape[0] // batch_size
        self.test.num_batch = _test.labels.shape[0] // batch_size
