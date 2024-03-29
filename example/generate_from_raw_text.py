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

with open('raw_text.txt', encoding='utf-8') as file:
    raw_text = list(map(str.strip, file))

wordcloud.generate_from_text(raw_text, random_state=42)

wordcloud.to_file('generate_from_raw_text.png')
