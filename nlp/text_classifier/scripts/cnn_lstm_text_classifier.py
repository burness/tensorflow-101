#-*-coding:utf-8-*-
from dataset_helpers.cut_doc import cutDoc
import numpy as np
from gensim import corpora,models
from pprint import pprint
import traceback
import sys
import cPickle as pickle
from config import *
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, MaxPooling1D, Embedding, Convolution1D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
# from keras.layers.merge import Concatenate
# from keras.engine.topology import Merge
from keras.optimizers import RMSprop,SGD

reload(sys)
sys.setdefaultencoding('utf-8')

class cnn_text_classifier:
    """ tf_idf_text_classifier: a text classifier of tfidf
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.dictionary = corpora.Dictionary()
        self.corpus = []
        self.labels = []
        self.cut_doc_obj = cutDoc()
        self.w2v_file = W2V_FILE
        self.class_num = CLASS_NUM
        self.filter_sizes = (3, 8)
        self.num_filters = 10
        self.hidden_dims = 64
    
    def load_word2vec(self):
        """ load_word2vec: load the w2v model
        """
        print "Start load word2vec model"
        self.w2vec = {}
        with open(self.w2v_file, "r") as fread:
            for line in fread.readlines():
                # print line
                line_list = line.strip().split(" ")
                word = line_list[0]
                word_vec = np.fromstring(' '.join(line_list[1:]), dtype=float, sep=' ')
                # print len(word_vec)
                self.w2vec[word] = word_vec
        print "Done load word2vec model"

    def __get_all_tokens_v2(self):
        """ get all tokens from file
        """
        print "load the tokens from file "
        with open(self.data_path.replace("all.csv","all_token.csv"), 'r') as fread:
            for line in fread.readlines():
                # print line
                try:
                    line_list = line.strip().split("\t")
                    label = line_list[0]
                    # print label
                    text_token = line_list[1].split("\\")
                    self.dictionary.add_documents([text_token])
                    self.labels.append(label)
                    self.corpus.append(text_token)
                except BaseException as e:
                    print e
                    continue
                
        # print "load dictionary fron file"
        # self.dictionary.load(self.data_path.replace("all.csv","cnn.dict"))

    def __get_all_tokens(self):
        """ get all tokens of the corpus
        """
        fwrite = open(self.data_path.replace("all.csv","all_token.csv"), 'w')
        with open(self.data_path, "r") as fread:
            i = 0
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
        print "save the dictionary"
        self.dictionary.save(self.data_path.replace("all.csv","cnn.dict"))
        self.num_data = i
        fwrite.close()

    def __filter_tokens(self, threshold_num=10):
        small_freq_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq < threshold_num ]
        self.dictionary.filter_tokens(small_freq_ids)
        self.dictionary.compactify()

    def gen_embedding_matrix(self, load4file=True):
        """ gen_embedding_matrix: generate the embedding matrix
        """
        if load4file:
            self.__get_all_tokens_v2()
        else:
            self.__get_all_tokens()
        print "before filter, the tokens len: {0}".format(self.dictionary.__len__())
        self.__filter_tokens()
        print "after filter, the tokens len: {0}".format(self.dictionary.__len__())
        self.sequence = []
        for file_token in self.corpus:
            temp_sequence = [x for x, y in self.dictionary.doc2bow(file_token)]
            self.sequence.append(temp_sequence)

        # bow_vec_file = open(self.data_path.replace("all.csv","bow_vec.pl"), 'wb')
        # print "after filter, the tokens len: {0}".format(self.dictionary.__len__())
        self.corpus_size = len(self.dictionary.token2id)
        self.embedding_matrix = np.zeros((self.corpus_size, EMBEDDING_DIM)) 
        print "corpus size: {0}".format(len(self.dictionary.token2id))
        for key, v in self.dictionary.token2id.items():
            key_vec = self.w2vec.get(key)
            if key_vec is not None:
                self.embedding_matrix[v] = key_vec
            else:
                self.embedding_matrix[v] = np.random.rand(EMBEDDING_DIM) - 0.5
    def step_decay(self, epoch):
        drop_every = 5
        decay_rate = (0.001*np.power(0.5, np.floor((1+drop_every)/drop_every))).astype('float32')
        return decay_rate

    def __build_network(self):
        embedding_layer = Embedding(self.corpus_size,
                            EMBEDDING_DIM,
                            weights=[self.embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
        # train a 1D convnet with global maxpooling
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Convolution1D(self.num_filters, 5, activation="relu")(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Convolution1D(self.num_filters, 5, activation="relu")(x)
        x = MaxPooling1D(5)(x)
        x = LSTM(64, dropout_W=0.2, dropout_U=0.2)(x)
        preds = Dense(self.class_num, activation='softmax')(x)
        print preds.get_shape()
        rmsprop = RMSprop(lr=0.01)
        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])

    def train(self):
        self.__split_train_test()
        self.__build_network()
        tensorboard = TensorBoard(log_dir="./logs/cnn_lstm/")
        ckpt_file = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        model_checkpoint = ModelCheckpoint(ckpt_file)
        # print self.train_set[:128]
        # print 'train labels: ',self.train_tag[:128]
        lrate = LearningRateScheduler(self.step_decay)
        self.model.fit(self.train_set, self.train_tag, validation_data=(self.test_set, self.test_tag),nb_epoch=100, batch_size=64, callbacks=[tensorboard, model_checkpoint,lrate])
        self.model.save(self.data_path.replace("all.csv","cnn.model"))

    def __split_train_test(self):
        # print len(self.sequence)
        # print self.sequence[0]
        self.data = pad_sequences(self.sequence, maxlen=MAX_SEQUENCE_LENGTH)
        #indices = np.arange(self.data.shape[0])
        #np.random.shuffle(indices)
        #self.data = self.data[indices]
        #self.labels = self.labels[indices]
        # print "__split_train_test {0}".format(self.data.shape)
        self.train_set, self.test_set, self.train_tag, self.test_tag = train_test_split(self.data, self.labels, test_size=0.2)
        print "train_tag {0}", ' '.join(self.train_tag)[0:1000]
        self.train_tag = to_categorical(np.asarray(self.train_tag))
        self.test_tag = to_categorical(np.asarray(self.test_tag))
        # print np.asarray(self.train_tag).shape




if __name__ == "__main__":
    cnn_text_classifier_obj = cnn_text_classifier("../data/origin_data/all.csv")
    cnn_text_classifier_obj.load_word2vec()
    cnn_text_classifier_obj.gen_embedding_matrix(load4file=True)
    cnn_text_classifier_obj.train()






