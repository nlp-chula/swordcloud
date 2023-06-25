import math
import numpy as np
from numpy.typing import NDArray
from typing import Union, Dict, List, Tuple, Literal, Optional
from random import Random
from sklearn.manifold import TSNE

#embedding EN
import gensim.downloader as api
#embedding TH
from pythainlp.word_vector import WordVector

from k_means_constrained import KMeansConstrained


def embed_w2v(
    word_counts: Union[Dict[str, int], Dict[str, float]],
    language: Literal['TH', 'EN']
) -> List[Tuple[str, NDArray[np.float32]]]:
    """
    Parameters
    ----------
    word_counts : dict from str to int
        contains words and associated frequency.

    lang : str, default = 'TH'
        language of input words, can be 'TH' or 'EN'
    
    Returns
    -------
    List of tuples of word and word vector ((str, list of float))
    """
    if language == 'TH':
        model = WordVector().get_model()
    else:
        # model = api.load('glove-wiki-gigaword-300') # This one is too big. Takes too long to load.
        model = api.load('glove-wiki-gigaword-50')

    return [(word, model[word]) for word in word_counts if word in model]

def plot_TSNE(
    model: List[Tuple[str, NDArray[np.float32]]],
    language: Literal['TH', 'EN'],
    random_state: Optional[Union[Random, int]]
):
    """
    Parameters
    ----------
    model : gensim.models.KeyedVector or list of tuple of (str, list[foat])
        word vector model (must come with 'labels') or list of tuple of word and word vectors (no 'labels' needed)

    labels : list of str (optional)
        words that we focused on; only in case of the 'model' is a whole word vector model.

    lang : str, default = 'TH'
        language of input words, can be 'TH' or 'EN'
    
    Returns
    -------
    Dict from str to tuple of floats, contains coordinates of words.
    """
    if random_state is None:
        random_state = Random()
    elif isinstance(random_state, int):
        random_state = Random(random_state)

    labels = list(map(lambda x: x[0], model))
    tokens = np.stack(list(map(lambda x: x[1], model)))

    max_possible_perplexity = tokens.shape[0] - 1
    if max_possible_perplexity < 1:
        raise ValueError('Too few words to plot')

    if language == 'TH':
        tsne_model = TSNE(
            n_components = 2,
            init = 'pca',
            n_iter = 2250,
            perplexity = min(7, max_possible_perplexity),
            early_exaggeration = 12,
            learning_rate = 210,
            random_state = random_state.randint(0, 1000)
        )
    else:
        tsne_model = TSNE(
            n_components = 2,
            init = 'pca',
            n_iter = 1000,
            perplexity = min(40, max_possible_perplexity),
            early_exaggeration = 12,
            learning_rate = 200,
            random_state = random_state.randint(0, 1000)
        )
    

    new_values = tsne_model.fit_transform(tokens)

    x: List[float] = []
    y: List[float] = []
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
    
    x_fab = 0
    y_fab = 0
    if min_x <= 0:
        x_fab = math.fabs(min_x)
    if min_y <= 0:
        y_fab = math.fabs(min_y)

    for value in new_values:
        x.append(value[0] + x_fab)
        y.append(value[1] + y_fab)

    return {labels[i]: (x[i], y[i]) for i in range(len(x))}

def kmeans_cluster(
    model: List[Tuple[str, NDArray[np.float32]]],
    n_clusters: int,
    random_state: Optional[Union[Random, int]],
    size_min: int = 10,
    size_max: int = 12
) -> List[int]:
    """
    Parameters
    ----------
    model : gensim.models.KeyedVector or list of tuple of (str, list[str])
        word vector model (must come with 'labels') or list of tuple of word and word vectors (no 'labels' needed)

    NUM_CLUSTERS : int (optional, default = 6)
        words that we focused on; only in case of the 'model' is a whole word vector model.

    lang : str, default = 'TH'
        language of input words, can be 'TH' or 'EN'
    
    Returns
    -------
    Dict from str to tuple of floats, contains coordinates of words.
    """
    if random_state is None:
        random_state = Random()
    elif isinstance(random_state, int):
        random_state = Random(random_state)

    X = list(map(lambda x: x[1], model))
    clf = KMeansConstrained(
        n_clusters = n_clusters,
        size_min = min(size_min, len(X) // n_clusters), # floor division
        size_max = max(size_max, -(len(X) // -n_clusters)), # ceiling division
        random_state = random_state.randint(0, 1000)
    )
    clf.fit_predict(X)
    return clf.labels_.tolist()
