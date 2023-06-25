# **swordcloud**
`swordcloud`: A semantic word cloud generator that uses t-SNE and k-means clustering to visualize words in high-dimensional semantic space. Based on [A. Mueller's `wordcloud` module](https://github.com/amueller/word_cloud), `swordcloud` can generate semantic word clouds from Thai and English texts, using `pythainlp` to tokenize Thai language data, and using t-SNE and k-means clustering to determine each word's position in the clouds.

## **Content**
1. [Installation](#installation)
2. [Example](#example)\
    2.1 [Initialize `SemanticWordCloud` instance](#initialize-semanticwordcloud-instance)\
    2.2 [Generate from Raw Text](#generate-from-raw-text)\
    2.3 [Generate from Word Frequencies](#generate-from-word-frequencies)\
    2.4 [Generate k-means Cluster Clouds](#generate-k-means-cluster-clouds)\
    2.5 [Recolor Words](#recolor-words)\
    2.6 [Export Word Clouds](#export-word-clouds)
3. [Documentation](#documentation)\
    3.1 [The `SemanticWordCloud` Class](#the-semanticwordcloud-class)\
    3.2 [Color "Functions"](#color-functions)

## **Installation**
`swordcloud` can be installed using `pip`:
```
pip install swordcloud
```
Optionally, if you want to be able to embed fonts directly into [the generated SVGs](#export-word-clouds), an `embedfont` extra can also be specified:
```
pip install swordcloud[embedfont]
```
As of **version 0.0.9**, the exact list of dependencies is as follow:
- `python >= 3.8`
- `numpy >= 1.21.0`
- `pillow`
- `matplotlib >= 1.5.3`
- `gensim >= 4.0.0`
- `pandas`
- `pythainlp >= 3.1.0`
- `k_means_constrained`
- `scikit-learn`
- (optional) `fonttools`

## **Example**
All code below can also be found in [the example folder](example).
### **Initialize `SemanticWordCloud` instance**
```python
from swordcloud import SemanticWordCloud
# See `Documentation` for detail about these color "functions"
from swordcloud.color_func import SingleColorFunc

wordcloud = SemanticWordCloud(
    language = 'TH'
    ...
    color_func = SingleColorFunc('black')
    ...
    random_state = 42
)
```
### **Generate from Raw Text**
```python
# Can also be one large string instead of a list of strings
raw_text = list(map(str.strip, open('raw_text.txt', encoding='utf-8')))

wordcloud.generate_from_text(raw_txt[:4000])
```
### **Generate from Word Frequencies**
```python
from collections import Counter

freq = Counter(
    word
    for line in open('pre_tokenized_text', encoding='utf-8')
        for word in line.strip().split('|')
)

wordcloud.generate_from_frequencies(freq)
```
### **Generate k-means Cluster Clouds**
```python
# Generate another 6 `SemanticWordCloud` instances where each cloud comes from k-means clustering
clouds = wordcloud.generate_from_text(raw_text[:4000], kmeans=6)
# Or directly from `generate_kmeans_cloud` if you already have word frequencies
clouds = wordcloud.generate_kmeans_cloud(freq, n_clusters=6)

# Each cloud can then be exported individually if needed
# See below for more detail on exporting word clouds
```
### **Recolor Words**
```python
# If the generated colors are not to your liking
# We can recolor them instead of re-generating the whole cloud
from swordcloud.color_func import ColorMapFunc
wordcloud.recolor(ColorMapFunc("magma"))
```
### **Export Word Clouds**
- As `pillow`'s `Image`
```python
img = wordcloud.to_image()
```
- As image file
```python
wordcloud.to_file('wordcloud.png')
```
- As SVG
```python
# Without embedded font
svg = wordcloud.to_svg()
# With embedded font
svg = wordcloud.to_svg(embed_font=True)

# Note that in order to be able to embed fonts
# the `fonttools` package needs to be installed
```
- As `numpy`'s image array
```python
array = wordcloud.to_array()
```

## **Documentation**
### **The `SemanticWordCloud` Class**
For most use cases, the `SemanticWordCloud` class is the main API the users will be interacting with.
```python
from swordcloud import SemanticWordCloud
wordcloud = SemanticWordCloud(...)
```
The `SemanticWordCloud` class has the following `__init__` arguments:
- a
- b

The `SemanticWordCloud` class has the following methods:
- a
- b

The `SemanticWordCloud` class has the following attributes:
- a
- b

### **Color "Functions"**
A number of built-in color "functions" can be accessed from  `swordcloud.color_func`:
```python
from swordcloud.color_func import <your_color_function_here>
```
The list of available functions is as follow:
- `ColorMapFunc` (default)\
    Return a random color from the user-specified [`matplotlib`'s colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html). This is the default behavior if the user does not provide the `color_func` argument to the `SemanticWordCloud` class. The default colormap used by `SemanticWordCloud` differs depending on the installed `matplotlib`'s version. See [src/swordcloud/wordcloud.py](src/swordcloud/wordcloud.py) for detail.
- `ImageColorFunc`\
    Use a user-provided colored image array to determine word color at each position on the canvas.
- `SingleColorFunc`\
    Always return the user-specified color every single time, resulting in every word having the same color.
- `ExactColorFunc`\
    Use a user-provided color dictionary to determine exactly which word should have which color.
- `RandomColorFunc`\
    Return a random color.

All the above functions, **except** `RandomColorFunc` which cannot be customized further, must be initialized before passing them to the `SemanticWordCloud` class. For example:
```python
from swordcloud.color_func import ColorMapFunc
color_func = ColorMapFunc("magma")
wordcloud = SemanticWordCloud(
    ...
    color_func = color_func
    ...
)
```
Users can also implement their own color functions, provided that they are callable with the following signature:

**Input**:
- `word: str`\
    The word we are coloring
- `font_size: int`\
    Font size of the word
- `position: tuple[int, int]`\
    Coordinate of the top-left point  of the word's bounding box on the canvas
- `orientation: int`\
    [`pillow`'s orientation](https://pillow.readthedocs.io/en/stable/reference/Image.html#transpose-methods).
- `font_path: str`\
    Path to the font file (OTF or TFF)
- `random_state: random.Random | int`\
    Python's `random.Random` or an `int` seed

The arguments should be in this exact order. However the function does not have to use all of them and every argument can also be `None`.

**Return**:\
Any object that can be interpreted as a color by `pillow`. See [`pillow`'s documentation](https://pillow.readthedocs.io/en/stable/) for more detail.