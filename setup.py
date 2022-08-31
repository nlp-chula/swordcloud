from setuptools import setup, find_packages

VERSION = '0.0.7' 
DESCRIPTION = 'Semantic word cloud package for Thai and English'
LONG_DESCRIPTION = 't-SNE and word embedding models for rearranging wordcloud'

# Setting up
setup(
        name="swordcloud", 
        version=VERSION,
        author="Attapol Thamrongrattanarit",
        author_email="<profte@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=['swordcloud'],
        package_data={'swordcloud': ['stopwords', 'thstopwords', 'THSarabun.ttf']},
        install_requires=['numpy>=1.6.1', 'pillow', 'matplotlib>=1.5.3', 'gensim>=4.0.0', 'pandas', 'pythainlp', 'k_means_constrained', 'sklearn'], 
        keywords=['python', 'word cloud', 't-SNE', 'K-means'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
