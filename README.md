# swordcloud
Semantic word cloud: A word cloud generator with the help of t-SNE and k-means clustering. **swordcloud**, based on [A. Mueller's word cloud module](https://github.com/amueller/word_cloud), can generate a semantic word cloud from Thai and English texts, using PyThaiNLP to tokenize Thai language data, and using t-SNE and k-means clustering to locate and group the words. <!-- Details can be found on the paper: (link to paper) -->

## Installation
using pip:
```
pip install swordcloud
```
this module requires Python 3, Numpy, Gensim 4.0 or later, PyThaiNLP, and Matplotlib

## Example
After installing and importing, you can generate the word cloud as in [example/sample.py](https://github.com/nlp-chula/swordcloud/blob/main/example/sample.py) has instructed. Noted that the input text for `generate_fron_text()` can be only `str` or a list of `str` only.

<!-- The generated semantic word clouds look like these: (img) (img) -->




