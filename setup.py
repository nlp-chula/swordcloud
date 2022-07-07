from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Semantic word cloud package for Thai and English'
LONG_DESCRIPTION = 't-SNE and word embedding models for rearranging wordcloud'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="swordcloud", 
        version=VERSION,
        author="Attapol Thamrongrattanarit",
        author_email="<profte@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        keywords=['python', 'word cloud'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
