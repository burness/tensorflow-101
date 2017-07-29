#-*-coding:utf-8-*-
from gensim.models import word2vec
# from config import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentence = word2vec.LineSentence(
    '../data/tag_day_ok.csv'
)
model = word2vec.Word2Vec(sentences=sentence, size=50, workers=4, min_count=5)
news_w2v = '../data/tag_word2vec.model'
model.save(news_w2v)
