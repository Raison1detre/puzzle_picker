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
    number_of_tile = 0 
    coefficient_for_find_number_of_the_tile = 0 # вычисляется как корень из (количество плиток/12)
    multiplier = 4 # зависит от угла поворота первой плитки относительно истины. Если угол 0/180 то 4, иначе 3. 

    def __init__(self, body, number):
        self.body = body
        self.smooth_body = None
        self.number_of_rotate = 0
        self.number = number
        self.sides = {1:None, 2:None, 3:None, 4:None}
        self.sides_matching_to_the_sides = {1:None, 2:None, 3:None, 4:None}
        self.rating_of_match = {1:None, 2:None, 3:None, 4:None}
        self.position_in_the_picture = 0
        self.flag = False #положение флага означает была ли повернута плитка относительно начального положения, и присвоили ли ей номер.
        
    def smooth(self,number_of_smoothing):
        body = self.body
        J = body.copy()
        for i in range(number_of_smoothing):
            J[1:-1] = (J[1:-1] // 2 + J[:-2] // 4 + J[2:] // 4)
            J[:, 1:-1] = (J[:, 1:-1] // 2 + J[:, :-2] // 4 + J[:, 2:] // 4)
        self.smooth_body = J

    def write_image(self, img, path):
        img = img
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
        a = similar_tiles[i,:].flatten()
        answer = np.asarray(a[1:],dtype=np.uint8)
        return answer


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
            similarity[0,count]=np.mean(np.abs(sim))
            similarity[1,count]=tile2.number
            similarity[2,count]=s2
            count +=1
        i,j = np.where(similarity == min(similarity[0,:]))
        if len(j) !=1:      #я без понятия почему, но np.where(similarity == s_min) иногда выдает больше одного ответа(с различными similarity). И если не отсекать лишние появляется ошибка. 
            j = j[0]
        answer = similarity[:,j].flatten()
        return answer
    
    def verification(self, pair1, pair2, list_of_tiles): # FIXME упростить вывод функции и написать комментарий
        """Функция используется для проверки найденных связей. Получает на вход два кортежа, pair1 и pair2, которые характеризуют две плитки, которые связаны с некоторой третьей плиткой, 
        и проверяет,связаны ли эти плитки с еще одной общей. list_of_tiles список хранящий все плитки.
        return =>> tuple(номер общей плитки(mutual_tile.number), номер соответствующей pair1 стороны плитки(mutual_tile.side)],номер стороны плитки(side_pair1),
        номер общей плитки(mutual_tile.number), номер соответствующей pair2 стороны плитки(mutual_tile.side)],номер стороны плитки(side_pair2)) или False если общей плитки найти не удалось.
        """
        s_p_1 = int(pair1[1]) #номер стороны которой соответствует pair1
        s_p_2 = int(pair2[1]) #номер стороны которой соответствует pair1

        ver_side_t1 = 4 if s_p_1 == 1 else s_p_1 - 1
        ver_side_t2 = 1 if s_p_2 == 4 else s_p_2 + 1
        v1 = self.find_similar_tile(list_of_tiles[int(pair1[0])],ver_side_t1,list_of_tiles)
        v2 = self.find_similar_tile(list_of_tiles[int(pair2[0])],ver_side_t2,list_of_tiles)
        if v1[0]==v2[0]:
            return (int(v1[0]), int(v1[1]), s_p_1, int(v2[0]), int(v2[1]), s_p_2)
        else:
            return False
    
    def rotate(self, tile, number_of_rotate):
        """Берет на вход плитку и целое число, означающее количество применений np.rot90() на эту плитку"""
        tile.body = np.rot90(tile.body, number_of_rotate)
        tile.smooth_body = np.rot90(tile.smooth_body,number_of_rotate)
        tile.sliser()
        tile.flag = True

    def find_number_the_tile_in_the_pickture(self,tile,list_of_tiles):
        tile_for_side_1 = self.find_similar_tile(tile, 1, list_of_tiles)
        tile_for_side_2 = self.find_similar_tile(tile, 2, list_of_tiles)
        tile_for_side_3 = self.find_similar_tile(tile, 3, list_of_tiles)
        tile_for_side_4 = self.find_similar_tile(tile, 4, list_of_tiles)

        result1 = self.verification(tile_for_side_1,tile_for_side_2, list_of_tiles)
        result2 = self.verification(tile_for_side_2,tile_for_side_3, list_of_tiles)
        result3 = self.verification(tile_for_side_3,tile_for_side_4, list_of_tiles)
        result4 = self.verification(tile_for_side_4,tile_for_side_1, list_of_tiles)
        """
        print("in process")
        print(tile_for_side_1[0])
        print(result1)
        print(result2)
        print(result3)
        print(result4)
        """
        if (result1 or result4) and list_of_tiles[tile_for_side_1[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_1[0]],tile_for_side_1[1]+1)
            list_of_tiles[tile_for_side_1[0]].position_in_the_picture = tile.position_in_the_picture + 1
            self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_1[0]],list_of_tiles)

        elif (result1 or result2) and list_of_tiles[tile_for_side_2[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_2[0]],tile_for_side_2[1])
            list_of_tiles[tile_for_side_2[0]].position_in_the_picture = tile.position_in_the_picture - tile.coefficient_for_find_number_of_the_tile * tile.multiplier
            self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_2[0]],list_of_tiles)

        elif (result2 or result3) and list_of_tiles[tile_for_side_3[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_3[0]],tile_for_side_3[1]+3)
            list_of_tiles[tile_for_side_3[0]].position_in_the_picture = tile.position_in_the_picture - 1
            self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_3[0]],list_of_tiles)

        elif (result3 or result4) and list_of_tiles[tile_for_side_4[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_4[0]],tile_for_side_4[1]+2)
            list_of_tiles[tile_for_side_4[0]].position_in_the_picture = tile.position_in_the_picture + tile.coefficient_for_find_number_of_the_tile * tile.multiplier
            self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_4[0]],list_of_tiles)

        else:
            print("done!")




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
    Tile.coefficient_for_find_number_of_the_tile = (len(list_of_tiles)/12)**0.5
    assembler = Assembler(list_of_tiles)
    first_tile = list_of_tiles[0]
    first_tile.flag = True
    assembler.find_number_the_tile_in_the_pickture(first_tile,list_of_tiles)
        

    for ex in list_of_tiles:
        print(ex.position_in_the_picture)
        print(ex.flag)

if __name__ == "__main__":
    solve()
