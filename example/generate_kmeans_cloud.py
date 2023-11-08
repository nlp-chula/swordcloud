from swordcloud import SemanticWordCloud
from swordcloud.color_func import FrequencyColorFunc

wordcloud = SemanticWordCloud(
    language = 'TH',
    width = 2400, # make sure the canvas is appropriately large for the number of clusters
    height = 1200,
    max_font_size = 150,
    prefer_horizontal = 1
)

with open('raw_text.txt', encoding='utf-8') as file:
    raw_text = list(map(str.strip, file))

wordcloud.generate_from_text(raw_text, kmeans=6, random_state=42, plot_now=False)

# Or if you already have a dictionary of word frequencies:
# wordcloud.generate_kmeans_cloud(freq, n_clusters=6, random_state=42, plot_now=False)

for sub_cloud, color in zip(wordcloud.sub_clouds, ["red", "blue", "brown", "green", "black", "orange"]):
    sub_cloud.recolor(FrequencyColorFunc(color), plot_now=False)

wordcloud.show()

# This will generate 6 png files
wordcloud.to_file('generate_kmeans_cloud.png')
