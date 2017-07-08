from keras.preprocessing.text import Tokenizer
import os
import operator
import sys
sys.path.append("..")
from config import *


class copus:
    def __init__(self, train_data_path, test_data_path, test_type="2013"):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.__build_dataset(test_type)

    def __build_dataset(self, test_type="2013"):
        self.train_source_file = os.path.join(self.train_data_path, TRAIN_X)
        self.train_target_file = os.path.join(self.train_data_path, TRAIN_Y)
        if test_type == "2013":
            self.test_source_file = os.path.join(self.test_data_path,
                                                 TEST_X_2013)
            self.test_target_file = os.path.join(self.test_data_path,
                                                 TEST_Y_2013)
        if test_type == "2014":
            self.test_source_file = os.path.join(self.test_data_path,
                                                 TEST_X_2014)
            self.test_target_file = os.path.join(self.test_data_path,
                                                 TEST_Y_2014)
        if test_type == "2015":
            self.test_source_file = os.path.join(self.test_data_path,
                                                 TEST_X_2015)
            self.test_target_file = os.path.join(self.test_data_path,
                                                 TEST_Y_2015)

    def read_copus_generator(self, batch_size=64):
        """ return a generator with the specified batch_size
        """
        logger.info("Beigin read copus {0}".format(file_name))
        data = []
        index = 0
        with open(file_name, 'r') as fread:
            while True:
                try:
                    line = fread.readline()
                    data.append(line)
                    index += 1
                    if index % 100000 == 0:
                        logger.info("The program has processed {0} lines ".
                                    format(index))
                except:
                    logger.info("Read End")
                    break
        tokenizer = Tokenizer(nb_words=30000)
        tokenizer.fit_on_texts(data)
        logger.info("word num: {0}".format(len(tokenizer.word_counts)))
        sorted_word_counts = sorted(
            tokenizer.word_counts.items(),
            key=operator.itemgetter(1),
            reverse=True)
        # save the word_counts to the meta
        with open(file_name.replace("train.", "meta."), "w") as fwrite:
            for word_cnt in sorted_word_counts:
                key = word_cnt[0]
                val = word_cnt[1]
                line = key + ":" + str(val) + "\n"
                fwrite.write(line)
        vectorize_data = tokenizer.texts_to_matrix(data)
        return vectorize_data


if __name__ == "__main__":
    copus_obj = copus("../../datasets/stanford/train",
                      "../../datasets/stanford/test")
    logger.info(copus_obj.train_source_data[0])
    logger.info("train copus shape {0}".format(
        copus_obj.train_source_data.shape))
