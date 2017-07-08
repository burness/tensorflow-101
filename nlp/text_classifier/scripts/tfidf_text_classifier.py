from dataset_helpers.cut_doc import cutDoc
import numpy as np
from gensim import corpora, models
from pprint import pprint
import traceback
import sys
import cPickle as pickle
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

reload(sys)
sys.setdefaultencoding('utf-8')


class tfidf_text_classifier:
    """ tf_idf_text_classifier: a text classifier of tfidf
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.dictionary = corpora.Dictionary()
        self.corpus = []
        self.labels = []
        self.cut_doc_obj = cutDoc()

    def __get_all_tokens(self):
        """ get all tokens of the corpus
        """
        fwrite = open(
            self.data_path.replace("all_title.csv", "all_token.csv"), 'w')
        with open(self.data_path, "r") as fread:
            i = 0
            # while True:
            for line in fread.readlines():
                try:
                    line_list = line.strip().split("\t")
                    label = line_list[0]
                    self.labels.append(label)
                    text = line_list[1]
                    text_tokens = self.cut_doc_obj.run(text)
                    self.corpus.append(' '.join(text_tokens))
                    self.dictionary.add_documents([text_tokens])
                    fwrite.write(label + "\t" + "\\".join(text_tokens) + "\n")
                    i += 1
                except BaseException as e:
                    msg = traceback.format_exc()
                    print msg
                    print "=====>Read Done<======"
                    break
        self.token_len = self.dictionary.__len__()
        print "all token len " + str(self.token_len)
        self.num_data = i
        fwrite.close()

    def __filter_tokens(self, threshold_num=10):
        small_freq_ids = [
            tokenid for tokenid, docfreq in self.dictionary.dfs.items()
            if docfreq < threshold_num
        ]
        self.dictionary.filter_tokens(small_freq_ids)
        self.dictionary.compactify()

    def vec(self):
        """ vec: get a vec representation of bow
        """
        self.__get_all_tokens()
        print "before filter, the tokens len: {0}".format(
            self.dictionary.__len__())
        vectorizer = CountVectorizer(min_df=1e-5)
        transformer = TfidfTransformer()
        # sparse matrix
        self.tfidf = transformer.fit_transform(
            vectorizer.fit_transform(self.corpus))
        words = vectorizer.get_feature_names()
        print "word len: {0}".format(len(words))
        # print self.tfidf[0]
        print "tfidf shape ({0},{1})".format(self.tfidf.shape[0],
                                             self.tfidf.shape[1])

        # write the tfidf vec into a file
        tfidf_vec_file = open(
            self.data_path.replace("all_title.csv", "tfidf_vec.pl"), 'wb')
        pickle.dump(self.tfidf, tfidf_vec_file)
        tfidf_vec_file.close()
        tfidf_label_file = open(
            self.data_path.replace("all_title.csv", "tfidf_label.pl"), 'wb')
        pickle.dump(self.labels, tfidf_label_file)
        tfidf_label_file.close()

    def split_train_test(self):
        self.train_set, self.test_set, self.train_tag, self.test_tag = train_test_split(
            self.tfidf, self.labels, test_size=0.2)
        print "train set shape: "
        print self.train_set.shape
        train_set_file = open(
            self.data_path.replace("all_title.csv", "tfidf_train_set.pl"),
            'wb')
        pickle.dump(self.train_set, train_set_file)
        train_tag_file = open(
            self.data_path.replace("all_title.csv", "tfidf_train_tag.pl"),
            'wb')
        pickle.dump(self.train_tag, train_tag_file)
        test_set_file = open(
            self.data_path.replace("all_title.csv", "tfidf_test_set.pl"), 'wb')
        pickle.dump(self.test_set, test_set_file)
        test_tag_file = open(
            self.data_path.replace("all_title.csv", "tfidf_test_tag.pl"), 'wb')
        pickle.dump(self.test_tag, test_tag_file)

    def train(self):
        print "Beigin to Train the model"
        lr_model = LogisticRegression()
        lr_model.fit(self.train_set, self.train_tag)
        print "End Now, and evalution the model with test dataset"
        # print "mean accuracy: {0}".format(lr_model.score(self.test_set, self.test_tag))
        y_pred = lr_model.predict(self.test_set)
        print classification_report(self.test_tag, y_pred)
        print confusion_matrix(self.test_tag, y_pred)
        print "save the trained model to tfidf_lr_model.pl"
        joblib.dump(lr_model,
                    self.data_path.replace("all_title.csv",
                                           "tfidf_lr_model.pl"))


if __name__ == "__main__":
    bow_text_classifier_obj = tfidf_text_classifier(
        "../data/origin_data/all_title.csv")
    bow_text_classifier_obj.vec()
    bow_text_classifier_obj.split_train_test()
    bow_text_classifier_obj.train()
