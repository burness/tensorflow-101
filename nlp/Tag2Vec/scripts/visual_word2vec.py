#-*-coding:utf-8-*-
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
model = gensim.models.Word2Vec.load("../data/tag_word2vec.model")
tag_id_name = {'UNK': "UNk"}
# tag_id=>tag_name
with open("../data/t_tag_infos.csv", "r") as fread:
    for line in fread.readlines():
        tag_id_name_list = line.split("\t")
        tag_id = tag_id_name_list[0]
        tag_name = tag_id_name_list[1].strip()
        tag_id_name[tag_id] = tag_name

tsne = TSNE(
    perplexity=30,
    n_components=2,
    init='pca',
    random_state=1,
    n_iter=5000,
    method='exact')


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)
    # plt.imshow()


X = model[model.wv.vocab]
X_tsne = tsne.fit_transform(X[:500])
labels = model.wv.vocab.keys()[:500]
labels = [tag_id_name[i].decode('utf-8') for i in labels]
plot_with_labels(X_tsne, labels)

tag_name_id = dict(zip(tag_id_name.values(), tag_id_name.keys()))


def get_topk(tag_word, model, topk=50):
    nearest_list = model.wv.similar_by_word(tag_name_id[tag_word], topn=topk)
    nearest_words = [tag_id_name[i[0]] for i in nearest_list]
    # nearest_words_score = [tag_id_name[i] for i in nearest_list]
    print "near the {0}, the top {1} words are {2}".format(
        tag_word, topk, ' '.join(nearest_words))


get_topk("知乎", model)
