# Derived from: amueller/wordcloud
# Author: Andreas Christian Mueller <t3kcit@gmail.com>
#
# (c) 2012
# Modified by: Paul Nechifor <paul@nechifor.net>
#
# License: MIT

import re
from base64 import b64encode
from io import BytesIO
from os import path, environ
from operator import itemgetter
from random import Random
from xml.sax import saxutils
from collections import Counter
from math import sqrt, ceil
from itertools import zip_longest
from typing import Dict, Tuple, List, Set, Union, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from PIL import Image, ImageDraw, ImageFont

from gensim.models import KeyedVectors
import gensim.downloader as api
from pythainlp.word_vector import WordVector

from .tokenization import process_tokens, unigrams_and_bigrams, word_tokenize
from .processing import plot_TSNE, kmeans_cluster
from .occupancy import IntegralOccupancyMap
from .color_func import RandomColorFunc, Color, ColorFunc

FILE = path.dirname(__file__)
DEFAULT_FONT_PATH = environ.get('FONT_PATH', path.join(FILE, 'THSarabun.ttf'))
DEFAULT_EN_STOPWORDS = set(map(str.strip, open(path.join(FILE, 'stopwords'), encoding="utf8")))
DEFAULT_TH_STOPWORDS = set(map(str.strip, open(path.join(FILE, 'thstopwords'), encoding="utf8")))

class SemanticWordCloud:
    """
    Semantic word cloud object for generating and drawing.

    Parameters
    ----------
    `language` : `str`
        Language of input text, can be 'TH' or 'EN'

    `model` : `gensim.models.KeyedVectors` or `str` (default=None)
        Word vector model. If None, the default model will be used,
        which is 'glove-wiki-gigaword-50' for English and 'thai2fit-wv' for Thai.

    `width` : `int` (default=400)
        Width of the canvas.

    `height` : `int` (default=200)
        Height of the canvas.

    `margin` : `int` (default=2)
        Margin of text boxes

    `scale` : `float` (default=1)
        Scaling between computation and drawing. For large word-cloud images,
        using scale instead of larger canvas size is significantly faster, but
        might lead to a coarser fit for the words.

    `prefer_horizontal` : `float` (default=0.9)
        The ratio of times to try horizontal fitting as opposed to vertical.
        If prefer_horizontal < 1, the algorithm will try rotating the word
        if it doesn't fit. (There is currently no built-in way to get only
        vertical words.)

    `color_func` : `Callable`, (default=None)
        Callable with parameters word, font_size, position, orientation,
        font_path, random_state that returns a PIL color for each word.

        See swordcloud.color_func for available built-in color functions.

    `background_color` : `Color` (default="white")
        Background color for the word cloud image.

    `mode` : `str` (default="RGB")
        Transparent background will be generated when mode is "RGBA" and
        background_color is None.

    `min_word_length` : `int`, default=0
        Minimum number of letters a word must have to be included.

    `max_words` : `int` (default=200)
        The maximum number of words.

    `stopwords` : `set[str]` (default=None)
        The words that will be eliminated. If None, the build-in STOPWORDS
        list will be used. Only used when using `generate_from_text` method.

    `font_path` : `str` (default=None)
        Path to the font file that will be used (OTF or TTF).
        If None, defaults to the FONT_PATH environment variable.
        If FONT_PATH is not available, uses THSarabun,
        which is bundled with swordcloud.

    `min_font_size` : `int` (default=4)
        Smallest font size to use. Will stop when there is no more room in this
        size.

    `max_font_size` : `int` (default=None)
        Maximum font size for the largest word. If None, height of the image is
        used.

    `font_step` : `int` (default=1)
        Step size for the font. font_step > 1 might speed up computation but
        give a worse fit.

    `relative_scaling` : `float` (default='auto')
        Importance of relative word frequencies for font-size. With
        relative_scaling=0, only word-ranks are considered.  With
        relative_scaling=1, a word that is twice as frequent will have twice
        the size.  If you want to consider the word frequencies and not only
        their rank, relative_scaling around .5 often looks good.
        If 'auto' it will be set to 0.5 unless repeat is true, in which
        case it will be set to 0.

    `regexp` : `str` (default=None)
        Regular expression to split the input text into tokens in process_text,
        overriding the default tokenizer. Only used when using `generate_from_text` method.

    `normalize_plurals` : `bool`, (default=True)
        For English only.
        Whether to remove trailing 's' from words. If True and a word
        appears with and without a trailing 's', the one with trailing 's'
        is removed and its counts are added to the version without
        trailing 's' -- unless the word ends with 'ss'.
        Only used when using `generate_from_text` method.

    `repeat` : `bool`, (default=False)
        Whether to repeat words and phrases until max_words or min_font_size
        is reached.

    `include_numbers` : `bool`, (default=False)
        Whether to include numbers as phrases or not.
        Only used when using `generate_from_text` method.

    `collocations` : `bool`, (default=False)
        Whether to include collocations (bigrams) of two words.
        Only used when using `generate_from_text` method.

    `collocation_threshold`: `int`, (default=30)
        Bigrams must have a Dunning likelihood collocation score greater than this
        parameter to be counted as bigrams. Default of 30 is arbitrary.

        See Manning, C.D., Manning, C.D. and Schütze, H., 1999. Foundations of
        Statistical Natural Language Processing. MIT press, p. 162
        https://nlp.stanford.edu/fsnlp/promo/colloc.pdf#page=22

    `random_state` : `random.Random` or `int` (default=None)
        Random state to be used for random drawing.

    Attributes
    ----------
    `sub_clouds` : `list[SemanticWordCloud]`
        List of sub clouds generated by K-Means algorithm.
        This attribute only exists after `generate_kmeans_cloud`
        or `generate_from_text` with `kmeans` is called.

    `words_` : `dict[str, float]`
        Word tokens with associated frequency.

    `layout_`: `list[tuple[tuple[string, float], int, tuple[int, int], int, color)]]`
        Encodes the fitted word cloud. For each word, it encodes the string, 
        normalized frequency, font size, position, orientation, and color.
        The frequencies are normalized by the most commonly occurring word.
        The color is in the format of 'rgb(R, G, B).'

    Notes
    -----
    Larger canvases with make the code significantly slower. If you need a
    large word cloud, try a lower canvas size, and set the scale parameter.

    The algorithm might give more weight to the ranking of the words
    than their actual frequencies, depending on the `max_font_size` and the
    scaling heuristic.
    """
    def __init__(
        self,
        language: Literal['TH', 'EN'],
        model: Optional[Union[KeyedVectors, str]] = None,
        width: int = 400,
        height: int = 200,
        margin: int = 2,
        scale: float = 1,
        prefer_horizontal: float = 0.9,
        color_func: ColorFunc = RandomColorFunc,
        background_color: Color = 'white',
        mode: str = "RGB",
        min_word_length: int = 0,
        max_words: int = 200,
        stopwords: Optional[Set[str]] = None,
        font_path: Optional[str] = None,
        min_font_size: int = 4,
        max_font_size: Optional[int] = None,
        font_step: int = 1,
        relative_scaling: Union[float, Literal['auto']] = 'auto',
        regexp: Optional[str] = None,
        normalize_plurals: Optional[bool] = None,
        repeat: bool = False,
        include_numbers: bool = False,
        collocations: bool = False,
        collocation_threshold: Optional[int] = None,
        random_state: Optional[Union[Random, int]] = None
    ):
        if language not in ('TH', 'EN'):
            raise ValueError(f"language must be either 'TH' or 'EN', got {repr(language)}.")
        self.language: Literal['TH', 'EN'] = language

        if isinstance(model, KeyedVectors):
            self.model = model
        elif isinstance(model, str):
            if language == 'EN':
                self.model = api.load(model)
            else:
                self.model = WordVector(model).get_model()
        elif language == 'TH':
            self.model = WordVector("thai2fit_wv").get_model()
        else:
            self.model = api.load("glove-wiki-gigaword-50")

        if normalize_plurals is None:
            self.normalize_plurals = language == 'EN'
        elif normalize_plurals and language == 'TH':
            self.normalize_plurals = False
            print(
                "WARNING: normalize_plurals is not supported for Thai "
                "since Thai does not have morphological plurals."
            )
        else:
            self.normalize_plurals = normalize_plurals

        if stopwords is not None:
            self.stopwords = {w.lower() for w in stopwords}
        elif language == 'TH':
            self.stopwords = DEFAULT_TH_STOPWORDS
        else:
            self.stopwords = DEFAULT_EN_STOPWORDS

        if collocation_threshold is not None:
            collocations = True
        else:
            collocation_threshold = 30
        if collocations and language == 'EN':
            collocations = False
            print(
                "WARNING: collocations is not supported for English "
                "since English word vector models do not contain bigrams."
            )
        self.collocations = collocations
        self.collocation_threshold = collocation_threshold

        if relative_scaling == "auto":
            if repeat:
                relative_scaling = 0
            else:
                relative_scaling = .5
        elif relative_scaling < 0 or relative_scaling > 1:
            raise ValueError(f"relative_scaling needs to be between 0 and 1, got {relative_scaling}.")
        self.relative_scaling = relative_scaling

        if regexp is not None:
            self.regexp = re.compile(regexp)
        elif regexp is None and language == 'EN':
            self.regexp = re.compile(r"\w[\w']*" if min_word_length <= 1 else r"\w[\w']+")
        else:
            self.regexp = None

        if not isinstance(random_state, Random):
            random_state = Random(random_state)
        self.random_state = random_state

        self.font_path = font_path or DEFAULT_FONT_PATH
        self.width = width
        self.height = height
        self.margin = margin
        self.prefer_horizontal = prefer_horizontal
        self.scale = scale
        self.max_words = max_words
        self.min_font_size = min_font_size
        self.font_step = font_step
        self.color_func = color_func
        self.background_color = background_color
        self.max_font_size = max_font_size
        self.mode = mode
        self.repeat = repeat
        self.include_numbers = include_numbers
        self.min_word_length = min_word_length

        self.sub_clouds: List[SemanticWordCloud] = []
        self.layout_: List[Tuple[Tuple[str, float], int, Tuple[int, int], Optional[int], Color]] = []

    def _embed_w2v(
        self,
        frequency_dict: Union[Dict[str, int], Dict[str, float]]
    ) -> List[Tuple[str, NDArray[np.float32]]]:
        """
        Parameters
        ----------
        `frequency_dict` : `dict[str, int]` | `dict[str, float]`
            contains words and associated frequency.

        Returns
        -------
        List of tuples of word and its word vector: `list[tuple[str, NDArray[float32]]]`
        """
        return [
            (word, self.model[word]) # type: ignore
            for word in sorted(frequency_dict, key=lambda k: frequency_dict[k], reverse=True)
                if word in self.model
        ][:self.max_words]

    def generate_from_frequencies(
        self,
        frequency_dict: Union[Dict[str, int], Dict[str, float]],
        max_font_size: Optional[int] = None,
        tsne_plot: Optional[Dict[str, Tuple[float, float]]] = None,
        plot_now: bool = True,
        random_state: Optional[Union[Random, int]] = None
    ):
        """
        Create a semantic word cloud from words and frequencies.

        Parameters
        ----------
        `frequency_dict` : `dict[str, int]`
            A contains words and associated frequency.

        `max_font_size` : `int` (default=None)
            Use this font-size instead of self.max_font_size

        `tsne_plot` : `dict[str, tuple[float, float]]` (default=None)
            A dictionary of word and its coordinates from TSNE plot.
            If None, a TSNE plot will be generated.

        `plot_now` : `bool` (default=True)
            Whether to plot the word cloud immediately.

        `random_state` : `random.Random` or `int` (default=None)
            Random state to be used for random drawing.
        """
        if len(frequency_dict) <= 0:
            raise ValueError(
                f"We need at least 1 word to plot a word cloud, got {len(frequency_dict)}."
            )

        if random_state is None:
            random_state = self.random_state
        elif not isinstance(random_state, Random):
            random_state = Random(random_state)

        if tsne_plot is None:
            tsne_plot = plot_TSNE(
                self._embed_w2v(frequency_dict),
                language = self.language,
                random_state = random_state
            )

        maxX = 0
        maxY = 0
        for ind in tsne_plot.values():
            if ind[0] > maxX:
                maxX = ind[0]
            if ind[1] > maxY:
                maxY = ind[1]
        
        # make sure frequencies are sorted and normalized
        frequencies = sorted(
            ((k, v) for k, v in frequency_dict.items() if k in tsne_plot),
            key=itemgetter(1),
            reverse=True
        )
        # largest entry will be 1
        max_frequency = float(frequencies[0][1])
        frequencies = [(word, freq / max_frequency) for word, freq in frequencies]

        height, width = self.height, self.width
        occupancy = IntegralOccupancyMap(height, width)

        # create image
        img_grey = Image.new("L", (width, height))
        draw = ImageDraw.Draw(img_grey)
        img_array = np.asarray(img_grey)

        font_sizes: List[int] = []
        positions: List[Tuple[int, int]] = []
        orientations: List[Optional[int]] = []
        colors: List[Color] = []

        if max_font_size is None:
            # if not provided use default font_size
            max_font_size = self.max_font_size

        if max_font_size is None:
            # figure out a good font size by trying to draw with
            # just the first two words
            if len(frequencies) == 1:
                # We only have one word. Make it big!
                font_size = self.height
            else:
                self.generate_from_frequencies(
                    dict(frequencies[:2]),
                    max_font_size = self.height,
                    plot_now = False,
                    tsne_plot = tsne_plot
                )
                # find font sizes
                sizes = [x[1] for x in self.layout_]
                try:
                    font_size = int(2 * sizes[0] * sizes[1] / (sizes[0] + sizes[1]))
                # quick fix for if self.layout_ contains less than 2 values
                # on very small images it can be empty
                except IndexError:
                    try:
                        font_size = sizes[0]
                    except IndexError:
                        raise ValueError(
                            "Couldn't find space to draw. The Canvas size might be too small."
                        )
        else:
            font_size = max_font_size

        self.words_ = dict(frequencies)

        if self.repeat and len(frequencies) < self.max_words:
            # pad frequencies with repeating words.
            times_extend = int(np.ceil(self.max_words / len(frequencies))) - 1
            # get smallest frequency
            frequencies_org = list(frequencies)
            downweight = frequencies[-1][1]
            for i in range(times_extend):
                frequencies.extend([
                    (word, freq * downweight ** (i + 1))
                    for word, freq in frequencies_org
                ])

        # start drawing grey image
        ## edit
        last_freq = 1.
        last_font_size = font_size
        cant_draw: Set[int] = set()
        for i, (word, freq) in enumerate(frequencies):
            if freq == 0:
                continue
            # select the font size
            rs = self.relative_scaling
            if rs != 0:
                font_size = int(round((rs * (freq / float(last_freq)) + (1 - rs)) * last_font_size))
            if random_state.random() < self.prefer_horizontal:
                orientation = None
            else:
                orientation = Image.ROTATE_90
            tried_other_orientation = False
            while True:
                # try to find a position
                font = ImageFont.truetype(self.font_path, font_size)
                # transpose font optionally
                transposed_font = ImageFont.TransposedFont(font, orientation=orientation)
                # get size of resulting text
                box_size = draw.textsize(word, font=transposed_font)

                # find possible places using integral image:
                resu = tsne_plot[word]
                to_mul_x = width / maxX
                to_mul_y = height / maxY
                fix_state = (to_mul_x * resu[0], to_mul_y * resu[1])
                result = occupancy.sample_position(
                    box_size[1] + self.margin, box_size[0] + self.margin, fix_state
                )
                
                if result is not None or font_size < self.min_font_size:
                    # either we found a place or font-size went too small
                    break
                # if we didn't find a place, make font smaller
                # but first try to rotate!
                if not tried_other_orientation and self.prefer_horizontal < 1:
                    orientation = Image.ROTATE_90
                    tried_other_orientation = True
                else:
                    font_size -= self.font_step
                    orientation = None

            if font_size < self.min_font_size:
                # Word became too small, skip to next word
                cant_draw.add(i)
                continue

            x, y = np.array(result) + self.margin // 2
            
            # actually draw the text
            draw.text((y, x), word, fill="white", font=transposed_font)
            positions.append((x, y))
            orientations.append(orientation)
            font_sizes.append(font_size)
            colors.append(
                self.color_func(
                    word = word,
                    font_size = font_size,
                    position = (x, y),
                    orientation = orientation,
                    random_state = random_state,
                    font_path = self.font_path
                )
            )

            # recompute integral image
            img_array = np.asarray(img_grey)
            # recompute bottom right
            occupancy.update(img_array, x, y)
            last_freq = freq
            last_font_size = font_size

        self.sub_clouds.clear()
        self.layout_ = list(zip(
            [(w, f) for i, (w, f) in enumerate(frequencies) if i not in cant_draw],
            font_sizes,
            positions,
            orientations,
            colors
        ))

        if plot_now:
            self.show()

    def _process_text(self, text: Union[str, List[str]]):
        """
        Tokenization, a.k.a. splits a long text into words, eliminates the stopwords.

        Parameters
        ----------
        `text` : `str` | `list[str]`
            The text to be processed.

        Returns
        -------
        `words` : `Counter[str]`
            Word tokens with associated frequency.
        """
        # EN is guaranteed to have regexp
        if self.regexp:
            if isinstance(text, list):
                text = '\n'.join(text)
            words: List[str] = self.regexp.findall(text)
            # remove 's
            if self.language == 'EN':
                words = [
                    word[:-2] if word.lower().endswith("'s") else word
                    for word in words
                ]
        else:
            if isinstance(text, str):
                words = word_tokenize(text)
            else:
                words = [word for t in text for word in word_tokenize(t)]

        def is_number(s: str):
            try:
                float(s)
                return True
            except ValueError:
                return False
        # remove numbers
        if not self.include_numbers:
            words = [word for word in words if not is_number(word)]
        # remove short words
        if self.min_word_length:
            words = [word for word in words if len(word) >= self.min_word_length]

        if self.collocations:
            word_counts = unigrams_and_bigrams(
                words, self.stopwords, self.collocation_threshold
            )
        else:
            # remove stopwords
            words = [word for word in words if word.lower() not in self.stopwords]
            if self.language == 'EN':
                word_counts = process_tokens(words, self.normalize_plurals)
            else:
                word_counts = Counter(words)

        return word_counts

    def generate_from_text(
        self,
        text: Union[str, List[str]],
        tsne_plot: Optional[Dict[str, Tuple[float, float]]] = None,
        kmeans: Optional[int] = None,
        plot_now: bool = True,
        random_state: Optional[Union[Random, int]] = None
    ):
        """
        Generate wordcloud from raw natural text.
        This method calls `process_text` and then either
        `generate_from_frequencies` or `generate_kmeans_cloud`.
        If kmeans is not None, the generated sub clouds can be
        accessed via `sub_clouds` attribute.

        Parameters
        ----------
        `text` : `str` | `list[str]`
            The text to be processed.

        `tsne_plot` : `dict[str, tuple[float, float]]` (default=None)
            A dictionary of word and its coordinates from TSNE plot.
            If None, a TSNE plot will be generated.
            Mutually exclusive with `kmeans`.

        `kmeans` : `int` (default=None)
            Number of clusters for K-Means algorithm.
            Mutually exclusive with `tsne_plot`.

        `plot_now` : `bool` (default=True)
            Whether to plot the word cloud immediately.

        `random_state` : `random.Random` or `int` (default=None)
            Random state to be used for random drawing.
        """
        if tsne_plot is not None and kmeans is not None:
            raise ValueError("Cannot use both tsne_plot and kmeans at the same time.")

        words = self._process_text(text)
        if kmeans is not None:
            if kmeans <= 1:
                raise ValueError(f'kmeans must be greater than 1, got {kmeans}.')
            self.generate_kmeans_cloud(
                frequency_dict = words,
                n_clusters = kmeans,
                plot_now = plot_now,
                random_state = random_state
            )
        else:
            self.generate_from_frequencies(
                frequency_dict = words,
                tsne_plot = tsne_plot,
                plot_now = plot_now,
                random_state = random_state
            )

    def _gen_kmeans_frequencies(
        self,
        model: List[Tuple[str, NDArray[np.float32]]],
        frequency_dict: Union[Dict[str, int], Dict[str, float]],
        n_clusters: int,
        random_state: Optional[Union[Random, int]] = None
    ):
        """
        Parameters
        ----------
        `model` : list[tuple[str, NDArray[float32]]]
            List of tuple of word and its word vectors

        `frequency_dict` : `dict[str, int]` | `dict[str, float]`
            Words and their associated frequency.

        `n_clusters` : `int`
            Number of clusters for K-Means algorithm.

        `random_state` : `random.Random` or `int` (default=None)
            Random state to be used for K-Means algorithm.

        Returns
        -------
        `list[tuple[int, list[tuple[str, float]]]]`
            List of tuple of cluster number and list of tuple of word and its frequency
        """
        if random_state is None:
            random_state = self.random_state
        elif not isinstance(random_state, Random):
            random_state = Random(random_state)

        label = list(map(lambda x: x[0], model))
        df = DataFrame(data={
            'word': label,
            'cluster': kmeans_cluster(
                model = model,
                n_clusters = n_clusters,
                random_state = random_state
            )
        })
        df['word_count'] = df['word'].map(frequency_dict)

        k_means_freq: List[Tuple[int, List[Tuple[str, float]]]] = []        
        for i in range(n_clusters):
            clus_i = df.loc[df['cluster'] == i]
            total = clus_i['word_count'].sum()
            clus_i_dict: Dict[str, float] = {}
            for _, row in clus_i.iterrows():
                clus_i_dict[row['word']] = row['word_count'] / total
            sorted_dict_i = sorted(clus_i_dict.items(), key=lambda item: item[1], reverse=True)

            lst: List[Tuple[str, float]] = []
            for k, v in sorted_dict_i:
                lst.append((k, v))
            k_means_freq.append((i, lst))

        return k_means_freq


    def generate_kmeans_cloud(
        self,
        frequency_dict: Union[Dict[str, int], Dict[str, float]],
        n_clusters: int,
        plot_now: bool = True,
        random_state: Optional[Union[Random, int]] = None
    ):
        """
        Generate semantic word cloud using K-Means algorithm.
        The generated sub clouds can be accessed via `sub_clouds` attribute.

        Parameters
        ----------
        `frequency_dict` : `dict[str, int]` | `dict[str, float]`
            Words and their associated frequency.

        `n_clusters` : `int`
            Number of clusters for K-Means algorithm.

        `plot_now` : `bool` (default=True)
            Whether to plot the word cloud immediately.

        `random_state` : `random.Random` or `int` (default=None)
            Random state to be used for K-Means algorithm.
        """
        if random_state is None:
            random_state = self.random_state
        elif not isinstance(random_state, Random):
            random_state = Random(random_state)

        model = self._embed_w2v(frequency_dict)
        kmeans_freq = self._gen_kmeans_frequencies(model, frequency_dict, n_clusters=n_clusters)

        n_vertical = n_horizontal = ceil(sqrt(n_clusters))
        if (n_vertical - 1) * n_horizontal >= n_clusters:
            n_vertical -= 1

        self.layout_.clear()
        self.sub_clouds = [
            SemanticWordCloud(
                language = self.language,
                model = self.model,
                width = self.width // n_horizontal,
                height = self.height // n_vertical,
                margin = self.margin,
                scale = self.scale,
                prefer_horizontal = self.prefer_horizontal,
                color_func = self.color_func,
                background_color = self.background_color,
                mode = self.mode,
                min_word_length = self.min_word_length,
                max_words = self.max_words,
                stopwords = self.stopwords,
                font_path = self.font_path,
                min_font_size = self.min_font_size,
                max_font_size = self.max_font_size,
                font_step = self.font_step,
                relative_scaling = self.relative_scaling,
                regexp = self.regexp.pattern if self.regexp else None,
                normalize_plurals = self.normalize_plurals,
                repeat = self.repeat,
                include_numbers = self.include_numbers,
                collocations = self.collocations,
                collocation_threshold = self.collocation_threshold,
                random_state = random_state,
            ) for _ in range(n_clusters)
        ]

        for i, cloud in enumerate(self.sub_clouds):
            topic_words = dict(kmeans_freq[i][1]) # list of (words, freq)
            cloud.generate_from_frequencies(topic_words, plot_now=False) # set topic

        if plot_now:
            self.show()

    def show(self):
        self._check_generated()
        if self.sub_clouds:
            n_clusters = len(self.sub_clouds)
            n_vertical = n_horizontal = ceil(sqrt(n_clusters))
            if (n_vertical - 1) * n_horizontal >= n_clusters:
                n_vertical -= 1

            dpi = plt.rcParams['figure.dpi']
            fig, axes = plt.subplots(
                n_vertical,
                n_horizontal,
                figsize=(self.width / dpi, self.height / dpi),
                sharex=True,
                sharey=True
            )
            for cloud, ax in zip_longest(self.sub_clouds, axes.flatten()):
                ax.axhline(y=0, color='black', linewidth=1)
                ax.axvline(x=0, color='black', linewidth=1)
                fig.add_subplot(ax)
                fig.tight_layout(rect=(0, 0.03, 1, 0.95))
                if cloud is not None:
                    plt.gca().imshow(cloud, aspect="auto", interpolation="bilinear")
                plt.gca().axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
        else:
            dpi = plt.rcParams['figure.dpi']
            plt.style.use('ggplot')
            plt.figure(figsize=(self.width / dpi, self.height / dpi))
            plt.imshow(self, interpolation="bilinear")
            plt.axis('off')
            plt.show()

    def _check_generated(self):
        """Check if `layout_` or `sub_clouds` was computed, otherwise raise error."""
        if not (self.layout_ or self.sub_clouds):
            raise ValueError("SemanticWordCloud has not been calculated, call generate first.")

    def to_image(self):
        """Convert to `pillow`'s image."""
        self._check_generated()
        if self.sub_clouds:
            raise ValueError(
                "Cannot directly convert a SemanticWordCloud with sub clouds to image. "
                "Call `to_image` individually from a sub cloud instead."
            )

        height, width = self.height, self.width

        img = Image.new(
            self.mode,
            (int(width * self.scale), int(height * self.scale)),
            self.background_color
        )
        draw = ImageDraw.Draw(img)
        for (word, _), font_size, position, orientation, color in self.layout_:
            font = ImageFont.truetype(self.font_path, int(font_size * self.scale))
            transposed_font = ImageFont.TransposedFont(font, orientation=orientation)
            pos = (int(position[1] * self.scale), int(position[0] * self.scale))
            draw.text(pos, word, fill=color, font=transposed_font)

        return img

    def recolor(
        self,
        color_func: Optional[ColorFunc] = None,
        plot_now: bool = True,
        random_state: Optional[Union[Random, int]] = None
    ):
        """
        Recolor existing layout.

        Applying a new coloring is much faster than generating the whole
        wordcloud.

        Parameters
        ----------
        `color_func` : `Callable`
            Function to generate new colors. If None, self.color_func is used again.

        plot_now : `bool` (default=True)
            Whether to plot the recolored word cloud immediately.
    
        `random_state` : `random.Random` or `int` (default=None)
            If None, use self.random_state.
            If not None, a fixed random state is used. If an int is given, this
            is used as seed for a random.Random state.
        """
        if random_state is None:
            random_state = self.random_state
        elif not isinstance(random_state, Random):
            random_state = Random(random_state)

        if color_func is None:
            color_func = self.color_func

        self._check_generated()
        if self.sub_clouds:
            for cloud in self.sub_clouds:
                cloud.recolor(color_func, plot_now=False, random_state=random_state)
        else:
            self.layout_ = [
                (
                    word_freq,
                    font_size,
                    position,
                    orientation,
                    color_func(
                        word = word_freq[0],
                        font_size = font_size,
                        position = position,
                        orientation = orientation,
                        random_state = random_state,
                        font_path = self.font_path
                    )
                )
                for word_freq, font_size, position, orientation, _ in self.layout_
            ]

        if plot_now:
            self.show()

    def to_file(self, filename: str):
        """
        Export to image file.

        Parameters
        ----------
        `filename` : `str`
            Location to write to.
        """
        if self.sub_clouds:
            filename, extension = path.splitext(filename)
            for i, cloud in enumerate(self.sub_clouds, start=1):
                cloud.to_file(filename + '_' + str(i) + extension)
        else:
            img = self.to_image()
            img.save(filename, optimize=True)

    def to_array(self):
        """
        Convert to numpy array.

        Returns
        -------
        `image` : `NDArray` size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return np.array(self.to_image())

    def __array__(self):
        """
        Convert to numpy array.

        Returns
        -------
        `image` : `NDArray` size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return self.to_array()

    def to_svg(
        self,
        embed_font: bool = False,
        optimize_embedded_font: bool = True,
        embed_image: bool = False
    ):
        """
        Export to SVG.

        Font is assumed to be available to the SVG reader. Otherwise, text
        coordinates may produce artifacts when rendered with replacement font.
        It is also possible to include a subset of the original font in WOFF
        format using `embed_font` (requires `fontTools`).

        Note that some renderers do not handle glyphs the same way, and may
        differ from `to_image` result. In particular, Complex Text Layout may
        not be supported. In this typesetting, the shape or positioning of a
        grapheme depends on its relation to other graphemes.

        Pillow, since version 4.2.0, supports CTL using ``libraqm``. However,
        due to dependencies, this feature is not always enabled. Hence, the
        same rendering differences may appear in ``to_image``. As this
        rasterized output is used to compute the layout, this also affects the
        layout generation. Use ``PIL.features.check`` to test availability of
        ``raqm``.

        Consistant rendering is therefore expected if both Pillow and the SVG
        renderer have the same support of CTL.

        Parameters
        ----------
        `embed_font` : `bool`, (default=False)
            Whether to include font inside resulting SVG file.

        `optimize_embedded_font` : `bool`, (default=True)
            Whether to be aggressive when embedding a font, to reduce size. In
            particular, hinting tables are dropped, which may introduce slight
            changes to character shapes (w.r.t. `to_image` baseline).

        `embed_image` : `bool`, (default=False)
            Whether to include rasterized image inside resulting SVG file.
            Useful for debugging.

        Returns
        -------
        `content` : `str`
            Word cloud image as SVG string
        """

        # TODO should add option to specify URL for font (i.e. WOFF file)

        # Make sure layout is generated
        self._check_generated()
        if self.sub_clouds:
            raise ValueError(
                "Cannot directly convert a SemanticWordCloud with sub clouds to SVG. "
                "Call `to_svg` from a sub cloud individually instead."
            )

        # Get output size, in pixels
        height, width = self.height, self.width

        # Get max font size
        if self.max_font_size is None:
            max_font_size = max(w[1] for w in self.layout_)
        else:
            max_font_size = self.max_font_size

        # Text buffer
        result: List[str] = []

        # Get font information
        font = ImageFont.truetype(self.font_path, int(max_font_size * self.scale))
        raw_font_family, raw_font_style = font.getname()
        # TODO properly escape/quote this name?
        font_family = repr(raw_font_family)
        # TODO better support for uncommon font styles/weights?
        raw_font_style = raw_font_style.lower()
        if 'bold' in raw_font_style:
            font_weight = 'bold'
        else:
            font_weight = 'normal'
        if 'italic' in raw_font_style:
            font_style = 'italic'
        elif 'oblique' in raw_font_style:
            font_style = 'oblique'
        else:
            font_style = 'normal'

        # Add header
        result.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * self.scale}" height="{height * self.scale}">'
        )

        # Embed font, if requested
        if embed_font:

            # Import here, to avoid hard dependency on fonttools
            import fontTools.subset
            from fontTools.ttLib import TTFont

            # Subset options
            options = fontTools.subset.Options(
                # Small impact on character shapes, but reduce size a lot
                hinting = not optimize_embedded_font,
                # On small subsets, can improve size
                desubroutinize = optimize_embedded_font,
                # Try to be lenient
                ignore_missing_glyphs = True,
            )

            # Load and subset font
            ttf = fontTools.subset.load_font(self.font_path, options)
            subsetter = fontTools.subset.Subsetter(options)
            characters = {c for item in self.layout_ for c in item[0][0]}
            text = ''.join(characters)
            subsetter.populate(text=text)
            subsetter.subset(ttf)

            # Export as WOFF
            # TODO is there a better method, i.e. directly export to WOFF?
            buffer = BytesIO()
            ttf.saveXML(buffer)
            buffer.seek(0)
            woff = TTFont(flavor='woff')
            woff.importXML(buffer)

            # Create stylesheet with embedded font face
            buffer = BytesIO()
            woff.save(buffer)
            data = b64encode(buffer.getbuffer()).decode('ascii')
            url = 'data:application/font-woff;charset=utf-8;base64,' + data
            result.append(
                f'<style>@font-face{{font-family:{font_family};font-weight:{font_weight};font-style:{font_style};src:url("{url}")format("woff");}}</style>'
            )

        # Select global style
        result.append(
            f'<style>text{{font-family:{font_family};font-weight:{font_weight};font-style:{font_style};}}</style>'
        )

        # Add background
        # if self.background_color is not None: # Should always be True
        result.append(
            f'<rect width="100%" height="100%" style="fill:{self.background_color}"></rect>'
        )

        # Embed image, useful for debug purpose
        if embed_image:
            image = self.to_image()
            data = BytesIO()
            image.save(data, format='JPEG')
            data = b64encode(data.getbuffer()).decode('ascii')
            result.append(
                f'<image width="100%" height="100%" href="data:image/jpg;base64,{data}"/>'
            )

        # For each word in layout
        for (word, _), font_size, (y, x), orientation, color in self.layout_:
            x *= self.scale
            y *= self.scale

            # Get text metrics
            font = ImageFont.truetype(self.font_path, int(font_size * self.scale))
            (size_x, _), (offset_x, offset_y) = font.font.getsize(word)
            ascent, _ = font.getmetrics()

            # Compute text bounding box
            min_x = -offset_x
            max_x = size_x - offset_x
            max_y = ascent - offset_y

            # Compute text attributes
            if orientation == Image.ROTATE_90:
                x += max_y
                y += max_x - min_x
                transform = f'translate({x},{y}) rotate(-90)'
            else:
                x += min_x
                y += max_y
                transform = f'translate({x},{y})'

            # Create node
            result.append(
                f'<text transform="{transform}" font-size="{font_size * self.scale}" style="fill:{color}">{saxutils.escape(word)}</text>'
            )

        # Complete SVG file
        result.append('</svg>')
        return '\n'.join(result)
