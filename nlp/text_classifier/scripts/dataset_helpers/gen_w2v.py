#-*-coding:utf-8-*-
from gensim.models import word2vec
# from config import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentence = word2vec.LineSentence(
    '/Users/burness/git_repository/dl_opensource/tensorflow-101/nlp/text_classifier/data/origin_data/all_token.csv'
)
model = word2vec.Word2Vec(sentences=sentence, size=50, workers=4, min_count=5)
# model.most_similar()
news_w2v = '/Users/burness/git_repository/dl_opensource/tensorflow-101/nlp/text_classifier/data/origin_data/news_w2v.model'
model.save(news_w2v)
# model.save
# model.wv.similar_by_word(u"习近平", topn=10)