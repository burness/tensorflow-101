from keras.optimizers import Adam, RMSprop, Nadam, Adadelta, SGD, Adagrad, Adamax
from ..config import *
from keras.layers import *


class TranslationModel():
    def __init__(self, optimizer_type, lr, loss_type):
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.loss_type = loss_type

    def setOptimizer(self):
        logger.info("Preparing optimizer: {0}, [LR: {1} - LOSS: {2}.]".format(
            self.optimizer_type, self.lr, self.loss_type))
        if self.optimizer_type.lower() == "sgd":
            self.optimizer = SGD(lr=self.lr, )
        elif self.optimizer_type.lower() == "rsmprop":
            self.optimizer = RMSprop(lr=self.lr)
        elif self.optimizer_type.lower() == "adagrad":
            self.optimizer = Adagrad(lr=self.lr)
        elif self.optimizer_type.lower() == "adam":
            self.optimizer = Adam(lr=self.lr)
        elif self.optimizer_type.lower() == "adamax":
            self.optimizer = Adamax(lr=self.lr)
        elif self.optimizer_type.lower() == "nadam":
            self.optimizer = Nadam(lr=self.lr)
        else:
            logger.info("\t WARNING: Not supported Now")

    def setLoss(self):
        pass

    def buildModel(self):
        src_text = Input(
            name="NMT_input", batch_shape=tuple([None, None]), dtype="int32")
