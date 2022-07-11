import math

import pandas as pd
from sklearn.manifold import TSNE

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# import matplotlib.font_manager as fm
# import numpy as np

#embedding EN
# from gensim.models import KeyedVectors, word2vec
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.test.utils import datapath, get_tmpfile
import gensim.downloader as api

#embedding TH
from pythainlp import word_vector

from k_means_constrained import KMeansConstrained


def embed_w2v(word_counts, lang='TH'):
    """
    Parameters
    ----------
    word_counts : dict from str to float
        contains words and associated frequency.

    lang : str, default = 'TH'
        language of input words, can be 'TH' or 'EN'
    
    Returns
    -------
    List of tuples of word and word vector ((str, list of float))
    """
    words = word_counts.keys()

    if lang=='TH':
      model = word_vector.get_model()
    else:
      model = api.load('glove-wiki-gigaword-300')

    # word2dict = {}
    # for word in words:
    #   if word in model.index_to_key:
    #     word2dict[word] = model[word]
    # word2vec = pd.DataFrame.from_dict(word2dict,orient='index')
    word2vecs = [(word, model[word]) for word in words if word in model.index_to_key]

    return word2vecs


def plot_TSNE(model,labels=None, lang='TH'):
    """
    Parameters
    ----------
    model : gensim.models.KeyedVector or list of tuple of (str, list[str])
        word vector model (must come with 'labels') or list of tuple of word and word vectors (no 'labels' needed)

    labels : list of str (optional)
        words that we focused on; only in case of the 'model' is a whole word vector model.

    lang : str, default = 'TH'
        language of input words, can be 'TH' or 'EN'
    
    Returns
    -------
    Dict from str to tuple of floats, contains coordinates of words.
    """
    if labels is None:
      labels = list(map(lambda x: x[0], model))
      tokens = list(map(lambda x: x[1], model))
    else:
      tokens = [model[word] for word in labels]


    if lang=='TH':
      tsne_model = TSNE(n_components=2, init='pca', n_iter=2250, perplexity=7, early_exaggeration = 12,
                        random_state=26,learning_rate=210)
    else:
      tsne_model = TSNE(n_components=2, init='pca', n_iter=1000, perplexity=40, early_exaggeration = 12,
                        random_state=23,learning_rate=200)
    

    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []               
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for value in new_values:
        if value[0] < min_x:
          min_x = value[0]
        if value[0] > max_x:
          max_x = value[0]
        if value[1] < min_y:
          min_y = value[1]
        if value[1] > max_y:
          max_y = value[1]
          
    if min_x <= 0:
        x_fab = math.fabs(min_x)

    if min_y <= 0:
        y_fab = math.fabs(min_y)

    for value in new_values:
        x.append(value[0] + x_fab)
        y.append(value[1] + y_fab)

    dic = {labels[i]:(x[i],y[i]) for i in range(len(x))}
    
    return dic

def generate_cluster_by_kmeans(model, NUM_CLUSTERS, size_min, size_max):

    X = list(map(lambda x: x[1], model))
    clf = KMeansConstrained(
        n_clusters=NUM_CLUSTERS,
        size_min=size_min,
        size_max=size_max,
        random_state=0
    )
    clf.fit_predict(X)
    clf.cluster_centers_
    grouped = clf.labels_.tolist()
    
    return grouped

def generate_kmeans_frequencies(model, label, word_count, NUM_CLUSTERS, size_min, size_max):
    df = pd.DataFrame(data={'word': label, 'cluster': generate_cluster_by_kmeans(model,NUM_CLUSTERS,size_min,size_max)})
    df['word_count'] = df['word'].map(word_count)
    k_means_freq = []
    
    for i in range(NUM_CLUSTERS):
        clus_i = df.loc[df['cluster'] == i]
        clus_i['total'] = clus_i['word_count'].sum()
        clus_i_dict = {}
        for _, row in clus_i.iterrows():
            clus_i_dict[row['word']] = row['word_count']/row['total']
        sorted_dict_i = sorted(clus_i_dict.items(), key=lambda item: item[1],reverse=True)[:10]

        lst = []
        for k,v in sorted_dict_i:
            lst.append((k,v))
        k_means_freq.append((i,lst))
    return k_means_freq

    
def rank_kmeans(kmeans_freq, rank_type='big'):
    if rank_type=='near':
        val = [1.0, 0.8, 0.6, 0.5, 0.4]
#     elif rank_type== 'big':
#         val = [1.0, 0.5, 0.25, 0.125, 0.1]
    else:
        val = [1.0, 0.5, 0.25, 0.125, 0.1]
    
    rank = {}
    for _,lst in kmeans_freq:
        rank.update(dict((tup[0], val[j]) if j < 4 else (tup[0], val[4]) for j,tup in enumerate(lst)))
        
    return rank