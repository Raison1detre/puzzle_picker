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

arr_of_likeness=[]
arr_of_map1=[]
list_of_white_side = []
list_of_side = []
list_of_number_side1=[]
list_of_number_side2=[]


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

def smooth(I):
    
    J = I.copy()
    
    J[1:-1] = (J[1:-1] // 2 + J[:-2] // 4 + J[2:] // 4)
    J[:, 1:-1] = (J[:, 1:-1] // 2 + J[:, :-2] // 4 + J[:, 2:] // 4)
    
    return J

def get_all_indexes(list_, element):
    return [index for index, value in enumerate(list_) if value == element]

tiles = [read_image(os.path.join("tiles", t)) for t in sorted(os.listdir("tiles"))]
dims = np.array([t.shape[:2] for t in tiles])
h, w = np.min(dims, axis=0)
x_nodes = np.arange(0, W, w)
y_nodes = np.arange(0, H, h)

tiles1 =[smooth(smooth(smooth(smooth(smooth(t))))) for t in tiles] 

for tile in range(len(tiles1)):
    for rot_t1 in range(4):
        for tile2 in range(len(tiles1)):
            for rot_t2 in range(4):
                if tile!=tile2:
                    t1=np.rot90(tiles1[tile],rot_t1)
                    t2=np.rot90(tiles1[tile2],rot_t2)
                    t1_slice =np.array(t1[:,299,0],dtype=np.int16)
                    t2_slice =np.array(t2[:,0,0],dtype=np.int16)
                    tile_likeness = np.mean(np.abs(t1_slice-t2_slice))
                    arr_of_likeness.append(tile_likeness)
                    list_of_number_side1.append([tile,rot_t1])
                    list_of_number_side2.append([tile2,rot_t2])


t1=np.rot90(tiles1[9],0)
t2=np.rot90(tiles1[10],0)
t1_slice =np.array(t1[:,0,0],dtype=np.int16)
t2_slice =np.array(t2[:,299,0],dtype=np.int16)
#print(np.mean(np.abs(t1_slice-t2_slice)))

#print(tiles[0][:,299,0])
#print(shape(tiles[0]))
#print(len(list_of_side))
#print(list_of_number_side1)
cur_match_side = []
side_for_cur_match_1 = []
side_for_cur_match_2 = []

#print(min(arr_of_likeness))


for i in range(len(tiles)):
    for j in range(4):
        index = get_all_indexes(list_of_number_side1,[i,j])
        index1 = get_all_indexes(list_of_number_side2,[i,j])
        index_all = index+index1
        s=[]
        for k in index_all:
            s.append(arr_of_likeness[k])

        
        #mn,mx = min(index),max(index)
        
        cur=arr_of_likeness.index(min(s))
        side_for_cur_match_1.append(list_of_number_side1[cur])
        side_for_cur_match_2.append(list_of_number_side2[cur])
        cur_match_side.append(arr_of_likeness[cur])


#print(cur_match_side)
#print('')
def find_four_pair(numb_tile):
    pair = []
    contact = []
    for i in range(4):
        if [numb_tile,i] in side_for_cur_match_1:
            i1 = side_for_cur_match_1.index([numb_tile,i])
            pair.append(i1)
            contact.append(side_for_cur_match_2[i1][0])
        elif[numb_tile,i] in side_for_cur_match_2:
            i2 = side_for_cur_match_2.index([numb_tile,i])
            pair.append(i2)
            contact.append(side_for_cur_match_1[i2][0])
    return (pair,contact)


all_pair=[]
all_contact=[]

for i in range(len(tiles)):
    p,c = find_four_pair(i)
    all_pair.append(p)
    all_contact.append(c)
#print(all_pair)
match1 = []
match2 = []
for til in range(len(tiles)):
    a = all_pair[til]
    for b in a:
        s1 = side_for_cur_match_1[b]
        s2 = side_for_cur_match_2[b]
        if s1 not in match1:
            match1.append(s1)
            match2.append(s2)

for i in range(len(match1)):
    t1 = tiles[match1[i][0]]
    t2 = tiles[match2[i][0]]
    t1_rot=np.rot90(t1,match1[i][1])
    t2_rot=np.rot90(t2,match2[i][1])
    t_stack = np.hstack((t1_rot,t2_rot))
    write_image(f"image_test{i}.ppm",t_stack)


 