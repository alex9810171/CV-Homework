import numpy as np
from PIL import Image

# input image and transform to binary
im_raw = Image.open("lena.bmp")
im_binary = im_raw.copy()
for i in range(im_binary.size[0]):
    for j in range(im_binary.size[1]):
        if(im_binary.getpixel((i, j)) < 128):
            im_binary.putpixel((i, j), 0)
        else:
            im_binary.putpixel((i, j), 255)
im_binary.save("lena_binary.bmp")
octogonal_kernel = np.array([[0, 255, 255, 255, 0],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [0, 255, 255, 255, 0]], dtype=int)

# Dilation
im_dilation = im_binary.copy()
for i in range(im_binary.size[0]):
    for j in range(im_binary.size[1]):
        if(im_binary.getpixel((i, j)) == 255):
            for k in range(i-2, i+2+1, 1):
                for l in range(j-2, j+2+1, 1):
                    if(k >= 0 and k < im_binary.size[0] and l >= 0 and l < im_binary.size[1] and octogonal_kernel[l-(j-2), k-(i-2)].item() == 255):
                        im_dilation.putpixel((k, l), 255)
im_dilation.save("lena_dilation.bmp")

# Erosion
im_erosion = im_binary.copy()
for i in range(im_binary.size[0]):
    for j in range(im_binary.size[1]):
        if(im_binary.getpixel((i, j)) == 255):
            if(i-2 < 0 or i+2 >= im_binary.size[0] or j-2 < 0 or j+2 >= im_binary.size[1]):
                im_erosion.putpixel((i, j), 0)
            else:
                delete = False
                for k in range(i-2, i+2+1, 1):
                    if(delete == True):
                        break
                    for l in range(j-2, j+2+1, 1):
                        if(octogonal_kernel[k-(i-2), l-(j-2)].item() == 255):
                            if(im_binary.getpixel((k, l)) != octogonal_kernel[l-(j-2), k-(i-2)].item()):
                                delete = True
                        if(delete == True):
                            break
                if(delete == True):
                    im_erosion.putpixel((i, j), 0)
im_erosion.save("lena_erosion.bmp")

# Opening
im_opening_temp = im_binary.copy()
for i in range(im_binary.size[0]):
    for j in range(im_binary.size[1]):
        if(im_binary.getpixel((i, j)) == 255):
            if(i-2 < 0 or i+2 >= im_binary.size[0] or j-2 < 0 or j+2 >= im_binary.size[1]):
                im_opening_temp.putpixel((i, j), 0)
            else:
                delete = False
                for k in range(i-2, i+2+1, 1):
                    if(delete == True):
                        break
                    for l in range(j-2, j+2+1, 1):
                        if(octogonal_kernel[l-(j-2), k-(i-2)].item() == 255):
                            if(im_binary.getpixel((k, l)) != octogonal_kernel[l-(j-2), k-(i-2)].item()):
                                delete = True
                        if(delete == True):
                            break
                if(delete == True):
                    im_opening_temp.putpixel((i, j), 0)
im_opening = im_opening_temp.copy()
for i in range(im_opening_temp.size[0]):
    for j in range(im_opening_temp.size[1]):
        if(im_opening_temp.getpixel((i, j)) == 255):
            for k in range(i-2, i+2+1, 1):
                for l in range(j-2, j+2+1, 1):
                    if(k >= 0 and k < im_opening_temp.size[0] and l >= 0 and l < im_opening_temp.size[1] and octogonal_kernel[l-(j-2), k-(i-2)].item() == 255):
                        im_opening.putpixel((k, l), 255)
im_opening.save("lena_opening.bmp")

# Closing
im_closing_temp = im_binary.copy()
for i in range(im_binary.size[0]):
    for j in range(im_binary.size[1]):
        if(im_binary.getpixel((i, j)) == 255):
            for k in range(i-2, i+2+1, 1):
                for l in range(j-2, j+2+1, 1):
                    if(k >= 0 and k < im_binary.size[0] and l >= 0 and l < im_binary.size[1] and octogonal_kernel[l-(j-2), k-(i-2)].item() == 255):
                        im_closing_temp.putpixel((k, l), 255)
im_closing = im_closing_temp.copy()
for i in range(im_closing_temp.size[0]):
    for j in range(im_closing_temp.size[1]):
        if(im_closing_temp.getpixel((i, j)) == 255):
            if(i-2 < 0 or i+2 >= im_closing_temp.size[0] or j-2 < 0 or j+2 >= im_closing_temp.size[1]):
                im_closing.putpixel((i, j), 0)
            else:
                delete = False
                for k in range(i-2, i+2+1, 1):
                    if(delete == True):
                        break
                    for l in range(j-2, j+2+1, 1):
                        if(octogonal_kernel[l-(j-2), k-(i-2)].item() == 255):
                            if(im_closing_temp.getpixel((k, l)) != octogonal_kernel[l-(j-2), k-(i-2)].item()):
                                delete = True
                        if(delete == True):
                            break
                if(delete == True):
                    im_closing.putpixel((i, j), 0)
im_closing.save("lena_closing.bmp")

# Hit-and-miss transform
l_hit_kernel = np.array([[255, 255],
                         [0, 255]], dtype=int)
l_miss_kernel = np.array([[255, 255],
                          [0, 255]], dtype=int)
im_hit_miss_a = im_binary.copy()
im_hit_miss_ac = im_binary.copy()
for i in range(im_hit_miss_ac.size[0]):
    for j in range(im_hit_miss_ac.size[1]):
        if(im_hit_miss_ac.getpixel((i, j)) == 255):
            im_hit_miss_ac.putpixel((i, j), 0)
        else:
            im_hit_miss_ac.putpixel((i, j), 255)

im_hit_miss_a_hit = im_hit_miss_a.copy()
im_hit_miss_ac_miss = im_hit_miss_ac.copy()
# hit
for i in range(im_hit_miss_a.size[0]):
    for j in range(im_hit_miss_a.size[1]):
        if(im_hit_miss_a.getpixel((i, j)) == 255):
            if(i-1 < 0 or j+1 >= im_hit_miss_a.size[1]):
                im_hit_miss_a_hit.putpixel((i, j), 0)
            else:
                delete = False
                for k in range(i-1, i+1, 1):
                    if(delete == True):
                        break
                    for l in range(j, j+1+1, 1):
                        if(l_hit_kernel[l-(j), k-(i-1)].item() == 255):
                            if(im_hit_miss_a.getpixel((k, l)) != l_hit_kernel[l-(j), k-(i-1)].item()):
                                delete = True
                        if(delete == True):
                            break
                if(delete == True):
                    im_hit_miss_a_hit.putpixel((i, j), 0) 

# miss
for i in range(im_hit_miss_ac.size[0]):
    for j in range(im_hit_miss_ac.size[1]):
        if(i+1 >= im_hit_miss_ac.size[0] or j-1 < 0):
            im_hit_miss_ac_miss.putpixel((i, j), 0)
        else:
            delete = False
            for k in range(i, i+1+1, 1):
                if(delete == True):
                   break
                for l in range(j-1, j+1, 1):
                    if(l_miss_kernel[l-(j-1), k-(i)].item() == 255):
                        if(im_hit_miss_ac.getpixel((k, l)) != l_miss_kernel[l-(j-1), k-(i)].item()):
                            delete = True
                    if(delete == True):
                        break
            if(delete == True):
                im_hit_miss_ac_miss.putpixel((i, j), 0)
            else:
                im_hit_miss_ac_miss.putpixel((i, j), 255)

# Intersection
im_hit_miss = im_binary.copy()
for i in range(im_hit_miss.size[0]):
    for j in range(im_hit_miss.size[1]):
        if(im_hit_miss_a_hit.getpixel((i, j)) == im_hit_miss_ac_miss.getpixel((i, j)) and im_hit_miss_a_hit.getpixel((i, j)) == 255):
            im_hit_miss.putpixel((i, j), 255)
        else:
            im_hit_miss.putpixel((i, j), 0)
im_hit_miss.save("lena_hit_miss.bmp")
