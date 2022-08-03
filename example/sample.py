# set path for data file

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)

# import the module
import swordcloud as swc
import pandas as pd

# read data and convert to a compatible data type
df = pd.read_csv('example/w_review_data.csv', names=['text'])
txt = list(df['text'][0:4000])

# Create a word cloud object
wn_wc = swc.WordCloud(background_color='white',
                     width=3360,
                     height=1890,
                     colormap='tab10',
                     max_font_size=200,
                     color_func=lambda *args, **kwargs: "black",
                     prefer_horizontal=1.0,
                     language='TH')

# 1.) plotting with t-SNE (default)
# wn_wc.generate_from_text(txt)

    # 1.1) for plot_now = None
    # wn_wc.generate_from_text(txt, plot_now=None)
    # plt.style.use('ggplot')
    # plt.figure(figsize=(9.6,4.8))
    # plt.imshow(wn_wc,interpolation="bilinear")
    # plt.axis('off')
    # plt.show()

# 2.) k-means plotting
wn_wc.generate_from_text(txt, kmeans=True)


