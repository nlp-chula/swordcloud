from collections import defaultdict, Counter
from itertools import tee
from math import log
from pythainlp.util.trie import Trie
from pythainlp.tokenize.newmm import segment
from typing import Sequence, Iterable, Set, Dict, Optional


def l(k: int, n: int, x: float):
    # dunning's likelihood ratio with notation from
    # http://nlp.stanford.edu/fsnlp/promo/colloc.pdf p162
    return log(max(x, 1e-10)) * k + log(max(1 - x, 1e-10)) * (n - k)


def score(count_bigram: int, count1: int, count2: int, n_words: int):
    """Collocation score"""
    if n_words <= count1 or n_words <= count2:
        # only one words appears in the whole document
        return 0
    N = n_words
    c12 = count_bigram
    c1 = count1
    c2 = count2
    p = c2 / N
    p1 = c12 / c1
    p2 = (c2 - c12) / (N - c1)
    score = l(c12, c1, p) + l(c2 - c12, N - c1, p) - l(c12, c1, p1) - l(c2 - c12, N - c1, p2)
    return -2 * score


def pairwise(iterable: Iterable[str]):
    # from itertool recipies
    # is -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unigrams_and_bigrams(
    words: Sequence[str],
    stopwords: Set[str],
    collocation_threshold: int,
):
    # We must create the bigrams before removing the stopword tokens from the words, or else we get bigrams like
    # "thank much" from "thank you very much".
    # We don't allow any of the words in the bigram to be stopwords
    bigrams = [p for p in pairwise(words) if not any(w.lower() in stopwords for w in p)]
    unigrams = [w for w in words if w.lower() not in stopwords]
    n_words = len(unigrams)

    counts_unigrams = Counter(unigrams)
    counts_bigrams = Counter(bigrams)
    # create a copy of counts_unigram so the score computation is not changed
    orig_counts = counts_unigrams.copy()

    # Include bigrams that are also collocations
    for bigram, count in counts_bigrams.items():
        bigram_string = ''.join(bigram)
        word1, word2 = bigram
        collocation_score = score(count, orig_counts[word1], orig_counts[word2], n_words)
        if collocation_score > collocation_threshold:
            # bigram is a collocation
            # discount words in unigrams dict. hack because one word might
            # appear in multiple collocations at the same time
            # (leading to negative counts)
            counts_unigrams[word1] -= counts_bigrams[bigram]
            counts_unigrams[word2] -= counts_bigrams[bigram]
            counts_unigrams[bigram_string] = counts_bigrams[bigram]
    for word, count in list(counts_unigrams.items()):
        if count <= 0:
            del counts_unigrams[word]
    return counts_unigrams


def process_tokens(words: Iterable[str], normalize_plurals: bool):
    """Normalize cases and remove plurals.

    Each word is represented by the most common case.
    If a word appears with an "s" on the end and without an "s" on the end,
    the version with "s" is assumed to be a plural and merged with the
    version without "s" (except if the word ends with "ss").

    Parameters
    ----------
    words : iterable of strings
        Words to count.

    normalize_plurals : bool, default=True
        Whether to try and detect plurals and remove trailing "s".

    Returns
    -------
    counts : dict from string to int
        Counts for each unique word, with cases represented by the most common
        case, and plurals removed.

    standard_forms : dict from string to string
        For each lower-case word the standard capitalization.
    """
    # words can be either a list of unigrams or bigrams
    # d is a dict of dicts.
    # Keys of d are word.lower(). Values are dicts
    # counting frequency of each capitalization
    d: Dict[str, Counter[str]] = defaultdict(Counter)
    for word in words:
        word_lower = word.lower()
        # get dict of cases for word_lower
        case_dict = d[word_lower]
        # increase this case
        case_dict[word] += 1
    if normalize_plurals:
        # merge plurals into the singular count (simple cases only)
        for key in list(d.keys()):
            if key.endswith('s') and not key.endswith("ss"):
                key_singular = key[:-1]
                if key_singular in d:
                    dict_plural = d[key]
                    dict_singular = d[key_singular]
                    for word, count in dict_plural.items():
                        singular = word[:-1]
                        dict_singular[singular] += count
                    del d[key]
    fused_cases: Counter[str] = Counter()
    for word_lower, case_dict in d.items():
        # Get the most popular case.
        first = case_dict.most_common(1)[0][0]
        fused_cases[first] = sum(case_dict.values())
    return fused_cases


def word_tokenize(text: str, custom_dict: Optional[Trie] = None, keep_whitespace: bool = False):

    """
    Word tokenizer.
    (override from PyThaiNLP.word_tokenize)

    Tokenizes running text into words (list of strings).

    :param str text: text to be tokenized

    :param pythainlp.util.Trie custom_dict: dictionary trie
    :param bool keep_whitespace: True to keep whitespaces, a common mark
                                 for end of phrase in Thai.
                                 Otherwise, whitespaces are omitted.
    :return: list of words
    """
    if custom_dict is not None:
        segments = segment(text, custom_dict)
    else:
        segments = segment(text)

    if not keep_whitespace:
        segments = [token.strip() for token in segments if token.strip()]

    return segments
