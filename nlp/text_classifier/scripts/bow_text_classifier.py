from dataset_helpers.cut_doc import cutDoc
import numpy as np
from gensim import corpora,models
from pprint import pprint
import traceback
import sys
import cPickle as pickle
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

reload(sys)
sys.setdefaultencoding('utf-8')

class bow_text_classifier:
    """ bow_text_classifier: a text classifier of bag of word
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
        fwrite = open(self.data_path.replace("all_title.csv","all_token.csv"), 'w')
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
                    self.corpus.append(text_tokens)
                    self.dictionary.add_documents([text_tokens])
                    fwrite.write(label+"\t"+"\\".join(text_tokens)+"\n")
                    i+=1
                except BaseException as e:
                    msg = traceback.format_exc()
                    print msg
                    print "=====>Read Done<======"
                    break
        self.token_len = self.dictionary.__len__()
        print "all token len "+ str(self.token_len)
        self.num_data = i
        fwrite.close()

    def __filter_tokens(self, threshold_num=10):
        small_freq_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq < threshold_num ]
        self.dictionary.filter_tokens(small_freq_ids)
        self.dictionary.compactify()

    def vec(self):
        """ vec: get a vec representation of bow
        """
        self.__get_all_tokens()
        print "before filter, the tokens len: {0}".format(self.dictionary.__len__())
        self.__filter_tokens()
        print "After filter, the tokens len: {0}".format(self.dictionary.__len__())
        self.bow = []
        for file_token in self.corpus:
            file_bow = self.dictionary.doc2bow(file_token)
            self.bow.append(file_bow)
        # write the bow vec into a file
        bow_vec_file = open(self.data_path.replace("all_title.csv","bow_vec.pl"), 'wb')
        pickle.dump(self.bow,bow_vec_file)
        bow_vec_file.close()
        bow_label_file = open(self.data_path.replace("all_title.csv","bow_label.pl"), 'wb')
        pickle.dump(self.labels,bow_label_file)
        bow_label_file.close()

    def to_csr(self):
        self.bow = pickle.load(open(self.data_path.replace("all_title.csv","bow_vec.pl"), 'rb'))
        self.labels = pickle.load(open(self.data_path.replace("all_title.csv","bow_label.pl"), 'rb'))
        data = []
        rows = []
        cols = []
        line_count = 0
        for line in self.bow:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        print "dictionary shape ({0},{1})".format(line_count, self.dictionary.__len__())
        bow_sparse_matrix = csr_matrix((data,(rows,cols)), shape=[line_count, self.dictionary.__len__()])
        print "bow_sparse matrix shape: "
        print bow_sparse_matrix.shape
        # rarray=np.random.random(size=line_count)
        self.train_set, self.test_set, self.train_tag, self.test_tag = train_test_split(bow_sparse_matrix, self.labels, test_size=0.2)
        print "train set shape: "
        print self.train_set.shape
        train_set_file = open(self.data_path.replace("all_title.csv","bow_train_set.pl"), 'wb')
        pickle.dump(self.train_set,train_set_file)
        train_tag_file = open(self.data_path.replace("all_title.csv","bow_train_tag.pl"), 'wb')
        pickle.dump(self.train_tag,train_tag_file)
        test_set_file = open(self.data_path.replace("all_title.csv","bow_test_set.pl"), 'wb')
        pickle.dump(self.test_set,test_set_file)
        test_tag_file = open(self.data_path.replace("all_title.csv","bow_test_tag.pl"), 'wb')
        pickle.dump(self.test_tag,test_tag_file)
    
    def train(self):
        print "Beigin to Train the model"
        lr_model = LogisticRegression()
        lr_model.fit(self.train_set, self.train_tag)
        print "End Now, and evalution the model with test dataset"
        # print "mean accuracy: {0}".format(lr_model.score(self.test_set, self.test_tag))
        y_pred = lr_model.predict(self.test_set)
        print classification_report(self.test_tag, y_pred)
        print confusion_matrix(self.test_tag, y_pred)
        print lr_model.score(self.test_set, self.test_tag)
        print "save the trained model to lr_model.pl"
        joblib.dump(lr_model, self.data_path.replace("all_title.csv","bow_lr_model.pl")) 


if __name__ == "__main__":
    bow_text_classifier_obj = bow_text_classifier("../data/origin_data/all_title.csv")
    bow_text_classifier_obj.vec()
    bow_text_classifier_obj.to_csr()
    # print len(bow_text_classifier_obj.train_set)
    # print len(bow_text_classifier_obj.test_set)
    bow_text_classifier_obj.train()
    # print bow_text_classifier_obj.all_words_len





