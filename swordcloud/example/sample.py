from pathlib import Path
import sys

from sklearn.cluster import k_means
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

import swordcloud as swc

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('swordcloud/example/w_review_data.csv', names=['text'])
txt = list(df['text'][0:4000])

wn_wc = swc.WordCloud(background_color='white',
                     width=3360,
                     height=1890,
                     colormap='tab10',
                     max_font_size=200,
                     color_func=lambda *args, **kwargs: "black",
                     prefer_horizontal=1.0,
                     language='TH')

wn_wc.generate_from_text(txt, kmeans=True)

# wn_wc_km.generate(txt, kmeans=True)

# plt.style.use('ggplot')
# plt.figure(figsize=(9.6,4.8))
# plt.imshow(wn_wc,interpolation="bilinear")
# plt.axis('off')
# plt.show()
