from math import fabs
import numpy as np
from numpy.typing import NDArray
from typing import Union, List, Tuple, Literal, Optional
from random import Random
from sklearn.manifold import TSNE
from k_means_constrained import KMeansConstrained

def plot_TSNE(
    model: List[Tuple[str, NDArray[np.float32]]],
    language: Literal['TH', 'EN'],
    random_state: Optional[Union[Random, int]]
):
    """
    Parameters
    ----------
    `model` : `list[tuple[str, NDArray[float32]]]`
        list of tuple of word and its word vectors
    `language` : `str`
        language of input words, can be 'TH' or 'EN'
    `random_state` : `Random` | `int`, default = None
        random state for TSNE model
    
    Returns
    -------
    `dict[str, tuple[float, float]]`, Coordinates of words in the TSNE plot
    """
    if not isinstance(random_state, Random):
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
            random_state = random_state.randint(0, 4294967295)
        )
    else:
        tsne_model = TSNE(
            n_components = 2,
            init = 'pca',
            n_iter = 1000,
            perplexity = min(40, max_possible_perplexity),
            early_exaggeration = 12,
            learning_rate = 200,
            random_state = random_state.randint(0, 4294967295)
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
        x_fab = fabs(min_x)
    if min_y <= 0:
        y_fab = fabs(min_y)

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
    `model` : `list[tuple[str, NDArray[float32]]]`
        list of tuple of word and its word vectors
    `n_clusters` : `int`
        Number of clusters to form
    `random_state` : `Random` | `int`, default = None
        random state for KMeans model
    `size_min` : `int`, default = 10
        Minimum number of words in each cluster
    `size_max` : `int`, default = 12
        Maximum number of words in each cluster
    
    Returns
    -------
    `list[int]`, each integer is the cluster label of each word in the 'model'
    """
    if not isinstance(random_state, Random):
        random_state = Random(random_state)

    X = list(map(lambda x: x[1], model))
    clf = KMeansConstrained(
        n_clusters = n_clusters,
        size_min = min(size_min, len(X) // n_clusters), # floor division
        size_max = max(size_max, -(len(X) // -n_clusters)), # ceiling division
        random_state = random_state.randint(0, 4294967295)
    )
    clf.fit_predict(X)
    return clf.labels_.tolist() # type: ignore
