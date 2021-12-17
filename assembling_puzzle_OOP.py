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
PATH = "C:\\Users\\alex\\my-py\\data\\0000_0000_0000\\tiles" # path to the folder of tiles (not folder of folders of tiles!)
NUMBER_OF_SMOOTHING = 5 

list_of_tiles = []

def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    return image

class Tile():
    number_of_false_connections = 0 # количество крайних сторон у крайних плиток которые ни с чем не связаны
    def __init__(self, body, number):
        self.body = body
        self.smooth_body = None
        self.number_of_rotate = 0
        self.number = number
        self.sides = {1:None, 2:None, 3:None, 4:None}
        self.sides_matching_to_the_sides = {1:None, 2:None, 3:None, 4:None}
        self.rating_of_match = {1:None, 2:None, 3:None, 4:None}

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

class Assembler():
    def __init__(self,list_of_tiles):
        dims = np.array([t.body.shape[:2] for t in list_of_tiles])
        h, w = np.min(dims, axis=0)
        x_nodes = np.arange(0, W, w)
        y_nodes = np.arange(0, H, h)
        self.matrix_of_ansvers = np.zeros((len(y_nodes),len(x_nodes),2), dtype=np.uint8)

    def fill_sides_matching_to_the_sides(self, list_of_tiles):
        """Получает на вход список всех плиток list_of_tiles. Для каждой плитки применяет find_similar_tile(tile,side,list_of_tiles) и по результатам 
        заполняет self.sides_matching_to_the_sides и self.rating_of_match, а также number_of_false_connections
        return =>> None"""
        for tile in list_of_tiles:
            for side in tile.sides_matching_to_the_sides:
                if tile.sides_matching_to_the_sides[side] == None:
                    tile_match = self.find_similar_tile(tile,side,list_of_tiles)
                    tile.sides_matching_to_the_sides[side] = (int(tile_match[0][1]),int(tile_match[0][2]))
                    tile.rating_of_match[side] = tile_match[0][0]
        Tile.number_of_false_connections = (((len(list_of_tiles)/12)**(0.5))*4*2)+(((len(list_of_tiles)/12)**(0.5))*3*2)
                
    def find_similar_tile(self, tile1, side_tile_1, list_of_tiles):
        """Получает на вход плитку tile1, номер её стороны side_tile_1, и список всех плиток. Работает в паре с check_similarity(tile1, side_tile_1, tile2).
        возвращает плитку и номер стороны, которая подходит к tile1, side_tile_1 наилучшим образом.
        return =>>  np.array [коэфф.соответствия(min), номер плитки(tile2.number), номер стороны плитки(tile2.side)] """
        similar_tiles = []
        for tile in list_of_tiles:
            if tile != tile1:
                similar_tiles.append(self.check_similarity(tile1,side_tile_1,tile))
        similar_tiles = np.asarray(similar_tiles)
        i,j = np.where(similar_tiles == min(similar_tiles[:,0]))
        j
        return similar_tiles[i,:]


    def check_similarity(self, tile1, side_tile_1, tile2):
        """Получает на вход плитку tile1 и номер её стороны side_tile_1, которую нужно сравнивает со всеми сторонами плитки tile2.
        Из 4 сторон выбирает ту с которой коэфф. соответствия меньше и
        return =>> np.array [коэфф.соответствия(min), номер плитки(tile2.number), номер стороны плитки(tile2.side)]"""
        similarity = np.zeros((3,4))
        count = 0
        for s2 in tile2.sides:
            s_1 = tile1.sides[side_tile_1]
            s_2 = tile2.sides[s2]
            s_2 = s_2[::-1]
            sim = s_1 - s_2
            similarity[0,count]=np.abs(np.mean(sim))
            similarity[1,count]=tile2.number
            similarity[2,count]=s2
            count +=1
        i,j = np.where(similarity == min(similarity[0,:]))
        if len(j) !=1:      #я без понятия почему, но np.where(similarity == s_min) иногда выдает больше одного ответа. И если не отсекать лишние появляется ошибка. 
            j = j[0]
        answer = similarity[:,j].flatten()
        return answer
    
    def verification(self, tile1, tile2, side_t1, side_t2, list_of_tiles):
        ver_side_t1 = 4 if side_t1==1 else side_t1-1
        ver_side_t2 = 1 if side_t2==4 else side_t2+1
        v1 = self.find_similar_tile(tile1,ver_side_t1,list_of_tiles)
        v2 = self.find_similar_tile(tile2,ver_side_t2,list_of_tiles)
        if v1[0][1]==v2[0][1]:
            return (v1,v2)
        else:
            return False

def print_image():
    for i in range(len(list_of_tiles)):
        til = list_of_tiles[i]
        til.write_image(f"image_test{i}.ppm")


def solve():
    for t in sorted(os.listdir(PATH)):
        tile = Tile(read_image(os.path.join(PATH, t)),int(t[0:4]))
        tile.smooth(NUMBER_OF_SMOOTHING)
        tile.sliser()
        list_of_tiles.append(tile)
    assembler = Assembler(list_of_tiles)
    #assembler.fill_sides_matching_to_the_sides(list_of_tiles)

    """ 
    for ti in list_of_tiles:
        print(ti.sides_matching_to_the_sides)
    count = 0
    for ti in list_of_tiles:
        for key in ti.rating_of_match:
            if ti.rating_of_match[key] != None:
                count +=1
        print(ti.rating_of_match)    
    """

    example_tile = list_of_tiles[0]
    pair1 = assembler.find_similar_tile(example_tile,int(4),list_of_tiles)
    pair2 = assembler.find_similar_tile(example_tile,int(1),list_of_tiles)
    side_pair1 = pair1[0][2]
    side_pair2 = pair2[0][2]
    print(pair1)
    print(pair2)
    result = assembler.verification(list_of_tiles[int(pair1[0][1])], list_of_tiles[int(pair2[0][1])], side_pair1, side_pair2, list_of_tiles)
    
    print(result)

    """
    print(Tile.number_of_false_connections)
    print(len(list_of_tiles))
    print(count)
    """
    #print_image()


if __name__ == "__main__":
    solve()
