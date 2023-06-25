import numpy as np
from PIL import ImageFont
from numpy.typing import NDArray
from random import Random
import matplotlib
from matplotlib.colors import Colormap
from PIL import ImageColor
from typing import Optional, Tuple, Dict, List, Any, Union, Callable

Color = Union[
    str,
    int,
    Tuple[int, int, int],
    Tuple[int, int, int, int]
]
ColorFunc = Callable[..., Color]

class ImageColorFunc:
    """Color generator based on a color image.

    Generates colors based on an RGB image. A word will be colored using
    the mean color of the enclosing rectangle in the color image.

    After construction, the object acts as a callable that can be passed as
    color_func to the word cloud constructor or to the recolor method.

    Parameters
    ----------
    image : nd-array, shape (height, width, 3)
        Image to use to generate word colors. Alpha channels are ignored.
        This should be the same size as the canvas. for the wordcloud.
    default_color : tuple or None, default=None
        Fallback colour to use if the canvas is larger than the image,
        in the format (r, g, b). If None, raise ValueError instead.
    """
    # returns the average color of the image in that region
    def __init__(self, image: NDArray[np.uint8], default_color: Optional[Tuple[int, int, int]] = None):
        if image.ndim not in [2, 3]:
            raise ValueError(f"ImageColorFunc needs an image with ndim 2 or 3, got {image.ndim}")
        if image.ndim == 3 and image.shape[2] not in [3, 4]:
            raise ValueError(f"A color image needs to have 3 or 4 channels, got {image.shape[2]}")
        self.image = image
        self.default_color = default_color

    def __call__(
        self,
        word: str,
        font_size: int,
        position: Tuple[int, int],
        orientation: Optional[int],
        font_path: str,
        *args: Any,
        **kwargs: Any
    ):
        """Generate a color for a given word using a fixed image."""
        # get the font to get the box size
        font = ImageFont.truetype(font_path, font_size)
        transposed_font = ImageFont.TransposedFont(font, orientation=orientation)
        # get size of resulting text
        box_size = transposed_font.getsize(word)
        x = position[0]
        y = position[1]
        # cut out patch under word box
        patch = self.image[x:x + box_size[0], y:y + box_size[1]]
        if patch.ndim == 3:
            # drop alpha channel if any
            patch = patch[:, :, :3]
        if patch.ndim == 2:
            raise NotImplementedError("Gray-scale images TODO")
        # check if the text is within the bounds of the image
        reshape = patch.reshape(-1, 3)
        if not np.all(reshape.shape):
            if self.default_color is None:
                raise ValueError(
                    "The provided image is smaller than the canvas. "
                    "Please provide default_color or resize the image."
                )
            return "rgb(%d, %d, %d)" % self.default_color
        color = np.mean(reshape, axis=0)
        return "rgb(%d, %d, %d)" % tuple(color)

class ColorMapFunc:
    """Color func created from matplotlib colormap.

    Parameters
    ----------
    colormap : string or matplotlib colormap
        Colormap to sample from

    Example
    -------
    >>> WordCloud(color_func=colormap_color_func("magma"))

    """
    def __init__(self, colormap: Optional[Union[Colormap, str]]):
        self.colormap = matplotlib.colormaps.get_cmap(colormap)

    def __call__(
        self,
        word: Optional[str] = None,
        font_size: Optional[int] = None,
        position: Optional[Tuple[int, int]] = None,
        orientation: Optional[int] = None,
        font_path: Optional[str] = None,
        random_state: Optional[Union[Random, int]] = None,
        *args: Any,
        **kwargs: Any
    ):
        if random_state is None:
            random_state = Random()
        elif isinstance(random_state, int):
            random_state = Random(random_state)
        r, g, b, _ = np.maximum(0, 255 * np.array(self.colormap(random_state.uniform(0, 1))))
        return f"rgb({r:.0f}, {g:.0f}, {b:.0f})"


def SingleColorFunc(color: str):
    """Create a color function which returns a single hue and saturation with.
    different values (HSV). Accepted values are color strings as usable by
    PIL/Pillow.

    >>> color_func1 = get_single_color_func('deepskyblue')
    >>> color_func2 = get_single_color_func('#00b4d2')
    """
    r, g, b, *_ = ImageColor.getrgb(color)
    def single_color_func(*args: Any, **kwargs: Any):
        return f"rgb({r:.0f}, {g:.0f}, {b:.0f})"
    return single_color_func

## from "examples/colored_by_group.py"
class ExactColorFunc:
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words: Dict[str, List[str]], default_color: str):
        self.word_to_color = {
            word: color
            for color, words in color_to_words.items()
                for word in words
        }

        self.default_color = default_color

    def __call__(self, word: str, *args: Any, **kwargs: Any):
        return self.word_to_color.get(word, self.default_color)

def RandomColorFunc(
    word: Optional[str] = None,
    font_size: Optional[int] = None,
    position: Optional[Tuple[int, int]] = None,
    orientation: Optional[int] = None,
    font_path: Optional[str] = None,
    random_state: Optional[Union[Random, int]] = None,
    *args: Any,
    **kwargs: Any
):
    """Random hue color generation.

    Default coloring method. This just picks a random hue with value 80% and
    lumination 50%.

    Parameters
    ----------
    word, font_size, position, orientation  : ignored.

    random_state : random.Random object or None, (default=None)
        If a random object is given, this is used for generating random
        numbers.

    """
    if random_state is None:
        random_state = Random()
    elif isinstance(random_state, int):
        random_state = Random(random_state)
    return "hsl(%d, 80%%, 50%%)" % random_state.randint(0, 255)