from swordcloud import SemanticWordCloud
from swordcloud.color_func import SingleColorFunc

wordcloud = SemanticWordCloud(
    language = 'TH',
    width = 2400, # make sure the canvas is appropriately large for the number of clusters
    height = 1200,
    max_font_size = 150,
    prefer_horizontal = 1,
    color_func = SingleColorFunc('black')
)

with open('raw_text.txt', encoding='utf-8') as file:
    raw_text = list(map(str.strip, file))

wordcloud.generate_from_text(raw_text, kmeans=6, random_state=42)

# Or if you already have a dictionary of word frequencies:
# wordcloud.generate_kmeans_cloud(freq, n_clusters=6, random_state=42)

# This will generate 6 png files
wordcloud.to_file('generate_kmeans_cloud.png')
