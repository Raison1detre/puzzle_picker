from typing import List
import numpy as np
import os
from numpy.core.fromnumeric import shape
from numpy.core.numeric import cross
from numpy.lib import rot90
from numpy.lib.function_base import append

from numpy.typing import _32Bit

W = 1200
H = 900
CHANNEL_NUM = 3  # we work with rgb images
MAX_VALUE = 255
PATH = "C:\\Users\\alex\\my-py\\tiles" # path to the folder of tiles (not folder of folders of tiles!)

class Tile():
    smooth_body = None
    side_1 = None
    side_2 = None
    side_3 = None
    side_4 = None
    side_matching_to_the_side_1 = None
    side_matching_to_the_side_2 = None
    side_matching_to_the_side_3 = None
    side_matching_to_the_side_4 = None
    number_of_rotate = 0


    def __init__(self, body, number):
        self.body = body
        self.number = number
    
    def smooth(self):
        body = self.body
        J = body.copy()
        J[1:-1] = (J[1:-1] // 2 + J[:-2] // 4 + J[2:] // 4)
        J[:, 1:-1] = (J[:, 1:-1] // 2 + J[:, :-2] // 4 + J[:, 2:] // 4)
        return J
    


def write_image(path, img):
    h, w = img.shape[:2]
    # ppm format requires header in special format
    header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
    with open(path, 'w') as f:
        f.write(header)
        for r, g, b in img.reshape((-1, CHANNEL_NUM)):
            f.write(f'{r} {g} {b} ')


def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    return image


