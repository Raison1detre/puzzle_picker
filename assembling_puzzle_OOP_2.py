from typing import List
import numpy as np
import os
from calculate_something import calculate_number_of_rotate

W = 1200
H = 900
CHANNEL_NUM = 3  # we work with rgb images
MAX_VALUE = 255
PATH = "C:\\Users\\alex\\my-py\\data\\0000_0000_0000\\tiles" # path to the folder of tiles (not folder of folders of tiles!)
NUMBER_OF_SMOOTHING = 5 
COEFFICIENT_OF_SINGLE_TON = 17 # коэффициент который определяет насколько однотонные стороны картинки будут выброшены из проверки
COEFFICIENT_OF_SIMILARITY = 13

list_of_tiles = []

def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.uint8).reshape((h, w, CHANNEL_NUM))
    return image

class Tile():
    number_of_tile = 0 
    coefficient_for_find_number_of_the_tile = 0 # вычисляется как корень из (количество плиток/12) при перемножении на multiplier дает количество плиток в строке
    multiplier = 4 # зависит от угла поворота первой плитки относительно истины. Если угол 0/180 то 4, иначе 3. 

    def __init__(self, body, number):
        self.body = body
        self.smooth_body = None
        self.number_of_rotate = 0
        self.number = number
        self.sides = {1:None, 2:None, 3:None, 4:None}
        self.sides_mask = {1:True, 2:True, 3:True, 4:True}
        self.sides_before_smooth = {1:None, 2:None, 3:None, 4:None}
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
        sb = self.body
        body = self.body
        h,w = sb.shape[:2]

        self.sides[1] = np.array(sb[:,w-1,:], dtype=np.int16)
        self.sides_before_smooth[1] = np.array(body[:,w-1,:], dtype=np.int16)
        if np.mean(np.std(self.sides_before_smooth[1],axis=0)) <  COEFFICIENT_OF_SINGLE_TON: # Это сделано чтобы монотонные стороны без выделяющихся элементов не могли пройти по цепочке дальше, я пока не знаю как фильтровать ошибку с крайними монотонными сторонами
            self.sides_mask[1] = False
        sb = np.rot90(sb,1)

        self.sides[4] = np.array(sb[:,h-1,:], dtype=np.int16)
        self.sides_before_smooth[4] = np.array(body[:,h-1,:], dtype=np.int16)
        if np.mean(np.std(self.sides_before_smooth[4],axis=0)) <  COEFFICIENT_OF_SINGLE_TON:
            self.sides_mask[4] = False
        sb = np.rot90(sb,1)

        self.sides[3] = np.array(sb[:,w-1,:], dtype=np.int16)
        self.sides_before_smooth[3] = np.array(body[:,w-1,:], dtype=np.int16)
        if np.mean(np.std(self.sides_before_smooth[3],axis=0)) <  COEFFICIENT_OF_SINGLE_TON:
            self.sides_mask[3] = False
        sb = np.rot90(sb,1)

        self.sides[2] = np.array(sb[:,h-1,:], dtype=np.int16)
        self.sides_before_smooth[2] = np.array(body[:,h-1,:], dtype=np.int16)
        if np.mean(np.std(self.sides_before_smooth[2],axis=0)) <  COEFFICIENT_OF_SINGLE_TON:
            self.sides_mask[2] = False
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
        return =>>  np.array [номер плитки(tile2.number), номер стороны плитки(tile2.side)] """
        similar_tiles = []
        for tile in list_of_tiles:
            if tile != tile1:
                similar_tiles.append(self.check_similarity(tile1,side_tile_1,tile))
        similar_tiles = np.asarray(similar_tiles)
        i,j = np.where(similar_tiles == min(similar_tiles[:,0]))
        j
        a = similar_tiles[i,:].flatten()
        if a[0] > COEFFICIENT_OF_SIMILARITY:
            return np.asarray([0,0],dtype=np.uint8)
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
            if tile1.sides_mask[side_tile_1] == False:  # Это сделано чтобы монотонные стороны не могли пройти по цепочке дальше, я пока не знаю как фильтровать ошибку с крайними монотонными сторонами
                similarity[0,count]=255
                similarity[1,count]=tile2.number
                similarity[2,count]=s2
            else:
                s_2_reverse = s_2[::-1]
                sim = s_1 - s_2_reverse
                similarity[0,count]=np.mean(np.abs(sim))
                similarity[1,count]=tile2.number
                similarity[2,count]=s2
            count +=1
        i,j = np.where(similarity == min(similarity[0,:]))
        if len(j) !=1:      #я без понятия почему, но np.where(similarity == s_min) иногда выдает больше одного ответа(с различными similarity). И если не отсекать лишние появляется ошибка. 
            j = j[0]
        answer = similarity[:,j].flatten()
        return answer
    
   
    def rotate(self, tile, number_of_rotate):
        """Берет на вход плитку и целое число, означающее количество применений np.rot90() на эту плитку"""
        tile.body = np.rot90(tile.body, number_of_rotate)
        tile.smooth_body = np.rot90(tile.smooth_body,number_of_rotate)
        tile.sliser()
        tile.flag = True
    
    def find_number_the_tile_in_the_pickture(self,tile,list_of_tiles):
        if tile.flag==False:
            print('flag false')
        tile_for_side = []
        for i in range(1,5):
            tile_for_side.append(self.find_similar_tile(tile, i, list_of_tiles))
            if tile_for_side[i-1][1] != 0:
                print(tile_for_side[i-1][0])
                self.rotate(list_of_tiles[tile_for_side[i-1][0]], calculate_number_of_rotate(i, tile_for_side[i-1][1]))
                if i == 1:
                    list_of_tiles[tile_for_side[i-1][0]].position_in_the_picture = tile.position_in_the_picture + 1
                elif i == 2:
                    list_of_tiles[tile_for_side[i-1][0]].position_in_the_picture = tile.position_in_the_picture - tile.coefficient_for_find_number_of_the_tile * tile.multiplier
                elif i == 3:
                    list_of_tiles[tile_for_side[i-1][0]].position_in_the_picture = tile.position_in_the_picture - 1
                elif i == 4:
                    list_of_tiles[tile_for_side[i-1][0]].position_in_the_picture = tile.position_in_the_picture + tile.coefficient_for_find_number_of_the_tile * tile.multiplier
        print("done!")




    '''
        if (result1 or result4) and list_of_tiles[tile_for_side_1[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_1[0]], calculate_number_of_rotate(1, tile_for_side_1[1]))
            
            pass
          #  self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_1[0]],list_of_tiles)

        elif (result1 or result2) and list_of_tiles[tile_for_side_2[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_2[0]], calculate_number_of_rotate(2, tile_for_side_2[1]))
            list_of_tiles[tile_for_side_2[0]].position_in_the_picture = tile.position_in_the_picture - tile.coefficient_for_find_number_of_the_tile * tile.multiplier
            pass  
          #  self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_2[0]],list_of_tiles)

        elif (result2 or result3) and list_of_tiles[tile_for_side_3[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_3[0]],calculate_number_of_rotate(3, tile_for_side_3[1]))
            list_of_tiles[tile_for_side_3[0]].position_in_the_picture = tile.position_in_the_picture - 1
            pass
          #  self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_3[0]],list_of_tiles)

        elif (result3 or result4) and list_of_tiles[tile_for_side_4[0]].flag == False:
            self.rotate(list_of_tiles[tile_for_side_4[0]],calculate_number_of_rotate(4, tile_for_side_4[1]))
            list_of_tiles[tile_for_side_4[0]].position_in_the_picture = tile.position_in_the_picture + tile.coefficient_for_find_number_of_the_tile * tile.multiplier
            pass
        #  self.find_number_the_tile_in_the_pickture(list_of_tiles[tile_for_side_4[0]],list_of_tiles)

        else:
            print("done!")
    '''



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
    Tile.coefficient_for_find_number_of_the_tile = int((len(list_of_tiles)/12)**0.5)
    assembler = Assembler(list_of_tiles)

    first_tile = list_of_tiles[0]
    assembler.rotate(first_tile,1)
    for instance in list_of_tiles:
        if instance.flag == True:
            assembler.find_number_the_tile_in_the_pickture(instance,list_of_tiles)
    
    for instance in list_of_tiles:
        if instance.flag == True:
            assembler.find_number_the_tile_in_the_pickture(instance,list_of_tiles)
    for instance in list_of_tiles:
        if instance.flag == True:
            assembler.find_number_the_tile_in_the_pickture(instance,list_of_tiles)        

    flags = []
    positions = []
    for ex in list_of_tiles:
        positions.append(ex.position_in_the_picture)
        flags.append(ex.flag)
    print(positions)
    print(flags)

    
if __name__ == "__main__":
    solve()
