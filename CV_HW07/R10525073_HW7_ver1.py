import numpy as np
from PIL import Image

# input image
im_raw = Image.open("lena.bmp")

# transform to binary
im_binary = im_raw.copy()
for j in range(im_binary.size[1]):
    for i in range(im_binary.size[0]):
        if(im_binary.getpixel((i, j)) < 128):
            im_binary.putpixel((i, j), 0)
        else:
            im_binary.putpixel((i, j), 255)

# downsample to 64x64 array
array_binary_downsample = np.zeros((im_binary.size[1]//8, im_binary.size[0]//8)).astype(int)
for j in range(im_binary.size[1]//8):
    for i in range(im_binary.size[0]//8):
        array_binary_downsample[j, i] = im_binary.getpixel((i*8, j*8))//255

# yokoi function
def count_yokoi(pos_x, pos_y, array_binary):
    q = 0
    r = 0
    # top-right
    if(pos_x+1 < array_binary.shape[1] and pos_y-1 >= 0 and array_binary[pos_y, pos_x+1] == 1
    and array_binary[pos_y-1, pos_x] == 1 and array_binary[pos_y-1, pos_x+1] == 1):
        r += 1
    elif(pos_x+1 < array_binary.shape[1] and array_binary[pos_y, pos_x+1] == 1):
        q += 1
    # top-left
    if(pos_x-1 >= 0 and pos_y-1 >= 0 and array_binary[pos_y-1, pos_x] == 1
    and array_binary[pos_y, pos_x-1] == 1 and array_binary[pos_y-1, pos_x-1] == 1):
        r += 1
    elif(pos_y-1 >= 0 and array_binary[pos_y-1, pos_x] == 1):
        q += 1
    # bottom-left
    if(pos_x-1 >= 0 and pos_y+1 < array_binary.shape[0] and array_binary[pos_y, pos_x-1] == 1
    and array_binary[pos_y+1, pos_x] == 1 and array_binary[pos_y+1, pos_x-1] == 1):
        r += 1
    elif(pos_x-1 >= 0 and array_binary[pos_y, pos_x-1] == 1):
        q += 1
    # bottom-right
    if(pos_x+1 < array_binary.shape[1] and pos_y+1 < array_binary.shape[0]
    and array_binary[pos_y+1, pos_x] == 1 and array_binary[pos_y, pos_x+1] == 1
    and array_binary[pos_y+1, pos_x+1] == 1):
        r += 1
    elif(pos_y+1 < array_binary.shape[0] and array_binary[pos_y+1, pos_x] == 1):
        q += 1
    if(r == 4):
        yokoi_number = 5
    else:
        yokoi_number = q
    return yokoi_number

equal = False
array_binary_new = array_binary_downsample.copy()
while equal == False:
    array_binary_old = array_binary_new.copy()
    array_yokoi_matrix = array_binary_old.copy()
    # step 1: count Yokoi connectivity number
    for j in range(array_binary_old.shape[0]):
        for i in range(array_binary_old.shape[1]):
            if(array_binary_old[j, i] == 1):
                array_yokoi_matrix[j, i] = count_yokoi(i, j, array_binary_old)
    # step 2: generate pair relationship matrix, for each pixel: p = 1, q = 2, background = 0
    array_pair_relationship_matrix = np.zeros((array_yokoi_matrix.shape[0], array_yokoi_matrix.shape[1])).astype(int)
    for j in range(array_yokoi_matrix.shape[0]):
        for i in range(array_yokoi_matrix.shape[1]):
            # check if self is edge and have an edge neighbor
            if(array_yokoi_matrix[j, i] == 1):
                if((i+1 < array_yokoi_matrix.shape[1] and array_yokoi_matrix[j, i+1] == 1)
                or (j-1 >= 0 and array_yokoi_matrix[j-1, i] == 1)
                or (i-1 >= 0 and array_yokoi_matrix[j, i-1] == 1)
                or (j+1 < array_yokoi_matrix.shape[0] and array_yokoi_matrix[j+1, i] == 1)):
                    array_pair_relationship_matrix[j, i] = 1
                else:
                    array_pair_relationship_matrix[j, i] = 2
    # step 3: if pair relationship operator is p, then check if connected shrink operator is removable (yokoi = 1)
    #         finally, delete those pixels satisfied above two conditions 
    for j in range(array_binary_new.shape[0]):
        for i in range(array_binary_new.shape[1]):
            if(array_pair_relationship_matrix[j, i] == 1):
                if(count_yokoi(i, j, array_binary_new) == 1):
                    array_binary_new[j, i] = 0
    equal = np.array_equal(array_binary_old, array_binary_new)

# save image
for j in range(array_binary_new.shape[0]):
    for i in range(array_binary_new.shape[1]):
        array_binary_new[j, i] = array_binary_new[j, i]*255
array_binary_new = array_binary_new.astype(np.uint8)
im_thin = Image.fromarray(array_binary_new)
im_thin.save("lena_thinning.bmp")
