import copy
import numpy as np
from PIL import Image

# pre-process image as array
im_raw = Image.open("lena.bmp")
im_array = np.array(im_raw)
im_array_size = im_array.shape

# test function
def showarray():
    print("initial array:")
    print(im_array)
    print("")
    print("upside down array:")
    print(im_array_upside_down)
    print("")
    print("left right array:")
    print(im_array_left_right)
    print("")
    print("diagonal array:")
    print(im_array_diagonal)

# flip upside down function
def flip_array(array):
    upside_down_array = np.zeros((im_array_size[0], im_array_size[1]))
    for i in range(im_array_size[1]):
        upside_down_array[i] = copy.deepcopy(array[im_array_size[1]-1-i])
    return upside_down_array

# flip left to right function
def fliplr_array(array):
    left_right_array = np.zeros((im_array_size[0], im_array_size[1]))
    for i in range(im_array_size[1]):
        left_right_array[i] = copy.deepcopy(array[i][::-1])
    return left_right_array

# flip & change data type from float to unit8
im_array_upside_down = flip_array(im_array).astype(np.uint8)
im_array_left_right = fliplr_array(im_array).astype(np.uint8)
im_array_diagonal = copy.deepcopy(im_array.T).astype(np.uint8)

# from array to image
im_upside_down = Image.fromarray(im_array_upside_down)
im_upside_down.save("lena_upside_down.bmp")
im_left_right = Image.fromarray(im_array_left_right)
im_left_right.save("lena_left_right.bmp")
im_diagonal = Image.fromarray(im_array_diagonal)
im_diagonal.save("lena_diagonal.bmp")

# part 2
im_rotate = im_raw.rotate(-45, expand=True).save("lena_rotate.bmp")
for i in range(im_raw.size[0]):
    for j in range(im_raw.size[1]):
        if(im_raw.getpixel((i, j)) < 128):
            im_raw.putpixel((i, j), 0)
        else:
            im_raw.putpixel((i, j), 255)
im_black_white = im_raw.save("lena_black_white.bmp")