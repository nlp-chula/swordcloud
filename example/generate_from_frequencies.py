from swordcloud import SemanticWordCloud
from swordcloud.color_func import SingleColorFunc

wordcloud = SemanticWordCloud(
    language = 'TH',
    width = 1600,
    height = 800,
    max_font_size = 150,
    prefer_horizontal = 1,
    color_func = SingleColorFunc('black')
)

with open("word_frequencies.tsv", encoding="utf-8") as file:
    freq = {}
    for line in file:
        word, count = line.strip().split('\t')
        freq[word] = int(count)

wordcloud.generate_from_frequencies(freq, random_state=42)

wordcloud.to_file('generate_from_frequencies.png')
