import numpy as np
from PIL import Image
from os import path, write

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

# Count Yokoi connectivity number
array_yokoi_matrix = array_binary_downsample.copy()
for j in range(array_binary_downsample.shape[0]):
    for i in range(array_binary_downsample.shape[1]):
        if(array_binary_downsample[j, i] == 1):
            q = 0
            r = 0
            # top-right
            if(i+1 < array_binary_downsample.shape[1] and j-1 >= 0 and array_binary_downsample[j, i+1] == 1
             and array_binary_downsample[j-1, i] == 1 and array_binary_downsample[j-1, i+1] == 1):
                r += 1
            elif(i+1 < array_binary_downsample.shape[1] and array_binary_downsample[j, i+1] == 1):
                q += 1
            # top-left
            if(i-1 >= 0 and j-1 >= 0 and array_binary_downsample[j-1, i] == 1
             and array_binary_downsample[j, i-1] == 1 and array_binary_downsample[j-1, i-1] == 1):
                r += 1
            elif(j-1 >= 0 and array_binary_downsample[j-1, i] == 1):
                q += 1
            # bottom-left
            if(i-1 >= 0 and j+1 < array_binary_downsample.shape[0] and array_binary_downsample[j, i-1] == 1
             and array_binary_downsample[j+1, i] == 1 and array_binary_downsample[j+1, i-1] == 1):
                r += 1
            elif(i-1 >= 0 and array_binary_downsample[j, i-1] == 1):
                q += 1
            # bottom-right
            if(i+1 < array_binary_downsample.shape[1] and j+1 < array_binary_downsample.shape[0]
             and array_binary_downsample[j+1, i] == 1 and array_binary_downsample[j, i+1] == 1
             and array_binary_downsample[j+1, i+1] == 1):
                r += 1
            elif(j+1 < array_binary_downsample.shape[0] and array_binary_downsample[j+1, i] == 1):
                q += 1
            if(r == 4):
                array_yokoi_matrix[j, i] = 5
            else:
                array_yokoi_matrix[j, i] = q

# output 64x64 matrix in txt
path = 'Yokoi_Matrix.txt'
with open(path, 'w') as file:
    for j in range(array_yokoi_matrix.shape[0]):
        for i in range(array_yokoi_matrix.shape[1]):
            if(array_yokoi_matrix[j, i] == 0):
                file.write(' ')
            else:
                file.write('%d' %(array_yokoi_matrix[j, i]))
        file.write('\n')
