from wordcloud import *

import numpy as np



def test_single_color_func():
    # test single color function for different color formats
    random = Random(42)

    red_function = get_single_color_func('red')
    assert red_function(random_state=None) == 'rgb(181, 0, 0)'

    hex_function = get_single_color_func('#00b4d2')
    assert hex_function(random_state=None) == 'rgb(0, 48, 56)'

    rgb_function = get_single_color_func('rgb(0,255,0)')
    assert rgb_function(random_state=None) == 'rgb(0, 107, 0)'

    rgb_perc_fun = get_single_color_func('rgb(80%,60%,40%)')
    assert rgb_perc_fun(random_state=None) == 'rgb(97, 72, 48)'

    hsl_function = get_single_color_func('hsl(0,100%,50%)')
    assert hsl_function(random_state=None) == 'rgb(201, 0, 0)'

