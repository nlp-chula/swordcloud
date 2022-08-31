from .color_from_image import ImageColorGenerator
from .wordcloud import WordCloud, STOPWORDS, STOPWORDS_TH, get_single_color_func, simple_grouped_color_func, grouped_color_func
from .processing import embed_w2v, plot_TSNE, kmeans_cluster, rank_kmeans
from .query_integral_image import query_integral_image
from .tokenization import process_tokens, unigrams_and_bigrams, score, pairwise, word_tokenize