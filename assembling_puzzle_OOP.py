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
NUMBER_OF_SMOOTHING = 5 

list_of_tiles = []

def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    return image

class Tile():
    smooth_body = None
    number_of_rotate = 0


    def __init__(self, body, number):
        self.body = body
        self.number = number
        self.sides = {1:None, 2:None, 3:None, 4:None}
        self.sides_matching_to_the_sides = {1:None, 2:None, 3:None, 4:None}

    def smooth(self,number_of_smoothing):
        body = self.body
        J = body.copy()
        for i in range(number_of_smoothing):
            J[1:-1] = (J[1:-1] // 2 + J[:-2] // 4 + J[2:] // 4)
            J[:, 1:-1] = (J[:, 1:-1] // 2 + J[:, :-2] // 4 + J[:, 2:] // 4)
        self.smooth_body = J

    def write_image(self, path):
        img = self.body
        h, w = img.shape[:2]
        # ppm format requires header in special format
        header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
        with open(path, 'w') as f:
            f.write(header)
            for r, g, b in img.reshape((-1, CHANNEL_NUM)):
                f.write(f'{r} {g} {b} ')
    
    def sliser(self):
        sb = self.smooth_body
        h,w = sb.shape[:2]
        self.sides[1] = np.array(sb[:,w-1,0], dtype=np.int16)
        sb = np.rot90(sb,1)
        self.sides[2] = np.array(sb[:,h-1,0], dtype=np.int16)
        sb = np.rot90(sb,1)
        self.sides[3] = np.array(sb[:,w-1,0], dtype=np.int16)
        sb = np.rot90(sb,1)
        self.sides[4] = np.array(sb[:,h-1,0], dtype=np.int16)
        sb = np.rot90(sb,1)

    
def check_similarity(side1,side2):
    similarity = []
    for s1 in side1.sides:
        for s2 in side2.sides:
            s_1 = side1.sides[s1]
            s_2 = side2.sides[s2]
            s_2 = s_2[::-1]
            sim = s_1 - s_2
            similarity.append(np.std(sim))
    print(min(similarity))


for t in sorted(os.listdir(PATH)):
    tile = Tile(read_image(os.path.join(PATH, t)),t)
    tile.smooth(NUMBER_OF_SMOOTHING)
    tile.sliser()
    list_of_tiles.append(tile)
    
for i in range(len(list_of_tiles)):
    til = list_of_tiles[i]
    til.write_image(f"image_test{i}.ppm")

check_similarity(list_of_tiles[0],list_of_tiles[3])
