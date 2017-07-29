from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(1)

# load final_embeddings
final_embeddings = pickle.load(open("../data/final_embeddings.model", "r"))
reverse_dictionary = pickle.load(open("../data/reverse_dictionary.dict", "r"))
dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))

# read from t_tag_infos.csv and save in a dict
tag_id_name = {'UNK': "UNk"}
with open("../data/t_tag_infos.csv", "r") as fread:
    for line in fread.readlines():
        tag_id_name_list = line.split("\t")
        tag_id = tag_id_name_list[0]
        tag_name = tag_id_name_list[1].strip()
        tag_id_name[tag_id] = tag_name


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


tsne = TSNE(
    perplexity=30,
    n_components=2,
    init='pca',
    random_state=1,
    n_iter=5000,
    method='exact')
plot_only = 500
print("final_embeddings size " + str(len(final_embeddings)))
valid_embeddings = final_embeddings[:plot_only, :]
valid_index = []
for index in xrange(plot_only):
    key = reverse_dictionary[index]
    if tag_id_name.has_key(key):
        valid_index.append(index)

low_dim_embs = tsne.fit_transform(valid_embeddings[valid_index])
labels = [
    tag_id_name[reverse_dictionary[i]].decode('utf-8') for i in valid_index
]
plot_with_labels(low_dim_embs, labels)


def get_topk(index, final_embeddings, k=10):
    print index
    presentation_labels = []
    similarity = np.matmul(final_embeddings, np.transpose(final_embeddings))
    nearest = (-similarity[index, :]).argsort()[1:10 + 1]
    print nearest
    for k in nearest:
        presentation_labels.append(tag_id_name[reverse_dictionary[k]])

    print "{0} nearest labels : {1}".format(
        tag_id_name[reverse_dictionary[index]], ' '.join(presentation_labels))


# 1000629
print dictionary['1000121']
get_topk(dictionary['1000121'], final_embeddings, k=10)