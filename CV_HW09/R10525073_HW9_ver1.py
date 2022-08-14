import math
import numpy as np
from PIL import Image

# input image
im_raw = Image.open("lena.bmp")

# Roberts operator
def Roberts_operator(image, threshold):
    array_image = np.array(image, dtype=int)
    array_image = np.pad(array_image, ((0, 1), (0, 1)), 'edge')
    edge_image = image.copy()
    for j in range(array_image.shape[0]-1):
        for i in range(array_image.shape[1]-1):
            r1 = array_image[j+1, i+1]-array_image[j, i]
            r2 = array_image[j+1, i]-array_image[j, i+1]
            gradient = (r1**2+r2**2)**0.5
            if(gradient < threshold):
                edge_image.putpixel((i, j), 255)
            else:
                edge_image.putpixel((i, j), 0)
    return edge_image

# Prewitt operator
def Prewitt_operator(image, threshold):
    array_image = np.array(image, dtype=int)
    array_image = np.pad(array_image, ((1, 1), (1, 1)), 'edge')
    edge_image = image.copy()
    for j in range(1, array_image.shape[0]-1):
        for i in range(1, array_image.shape[1]-1):
            p1 = (array_image[j+1, i-1]+array_image[j+1, i]+array_image[j+1, i+1])-(array_image[j-1, i-1]+array_image[j-1, i]+array_image[j-1, i+1])
            p2 = (array_image[j-1, i+1]+array_image[j, i+1]+array_image[j+1, i+1])-(array_image[j-1, i-1]+array_image[j, i-1]+array_image[j+1, i-1])
            gradient = (p1**2+p2**2)**0.5
            if(gradient < threshold):
                edge_image.putpixel((i-1, j-1), 255)
            else:
                edge_image.putpixel((i-1, j-1), 0)
    return edge_image

# Sobel operator
def Sobel_operator(image, threshold):
    array_image = np.array(image, dtype=int)
    array_image = np.pad(array_image, ((1, 1), (1, 1)), 'edge')
    edge_image = image.copy()
    for j in range(1, array_image.shape[0]-1):
        for i in range(1, array_image.shape[1]-1):
            s1 = (array_image[j+1, i-1]+2*array_image[j+1, i]+array_image[j+1, i+1])-(array_image[j-1, i-1]+2*array_image[j-1, i]+array_image[j-1, i+1])
            s2 = (array_image[j-1, i+1]+2*array_image[j, i+1]+array_image[j+1, i+1])-(array_image[j-1, i-1]+2*array_image[j, i-1]+array_image[j+1, i-1])
            gradient = (s1**2+s2**2)**0.5
            if(gradient < threshold):
                edge_image.putpixel((i-1, j-1), 255)
            else:
                edge_image.putpixel((i-1, j-1), 0)
    return edge_image

# Frei and Chen operator
def FC_operator(image, threshold):
    array_image = np.array(image, dtype=int)
    array_image = np.pad(array_image, ((1, 1), (1, 1)), 'edge')
    edge_image = image.copy()
    for j in range(1, array_image.shape[0]-1):
        for i in range(1, array_image.shape[1]-1):
            f1 = (array_image[j+1, i-1]+(2**0.5)*array_image[j+1, i]+array_image[j+1, i+1])-(array_image[j-1, i-1]+(2**0.5)*array_image[j-1, i]+array_image[j-1, i+1])
            f2 = (array_image[j-1, i+1]+(2**0.5)*array_image[j, i+1]+array_image[j+1, i+1])-(array_image[j-1, i-1]+(2**0.5)*array_image[j, i-1]+array_image[j+1, i-1])
            gradient = (f1**2+f2**2)**0.5
            if(gradient < threshold):
                edge_image.putpixel((i-1, j-1), 255)
            else:
                edge_image.putpixel((i-1, j-1), 0)
    return edge_image

# Kirsch compass operator
def Kirsch_operator(image, threshold):
    array_image = np.array(image, dtype=int)
    array_image = np.pad(array_image, ((1, 1), (1, 1)), 'edge')
    edge_image = image.copy()
    for j in range(1, array_image.shape[0]-1):
        for i in range(1, array_image.shape[1]-1):
            k = np.zeros(8, dtype=int)
            k[0] = (-3)*(array_image[j-1, i-1]+array_image[j-1, i]+array_image[j, i-1]+array_image[j+1, i-1]+array_image[j+1, i])+5*(array_image[j-1, i+1]+array_image[j, i+1]+array_image[j+1, i+1])
            k[1] = (-3)*(array_image[j-1, i-1]+array_image[j, i-1]+array_image[j+1, i-1]+array_image[j+1, i]+array_image[j+1, i+1])+5*(array_image[j-1, i]+array_image[j-1, i+1]+array_image[j, i+1])
            k[2] = (-3)*(array_image[j, i-1]+array_image[j, i+1]+array_image[j+1, i-1]+array_image[j+1, i]+array_image[j+1, i+1])+5*(array_image[j-1, i-1]+array_image[j-1, i]+array_image[j-1, i+1])
            k[3] = (-3)*(array_image[j-1, i+1]+array_image[j, i+1]+array_image[j+1, i-1]+array_image[j+1, i]+array_image[j+1, i+1])+5*(array_image[j-1, i-1]+array_image[j-1, i]+array_image[j, i-1])
            k[4] = (-3)*(array_image[j-1, i]+array_image[j-1, i+1]+array_image[j, i+1]+array_image[j+1, i]+array_image[j+1, i+1])+5*(array_image[j-1, i-1]+array_image[j, i-1]+array_image[j+1, i-1])
            k[5] = (-3)*(array_image[j-1, i-1]+array_image[j-1, i]+array_image[j-1, i+1]+array_image[j, i+1]+array_image[j+1, i+1])+5*(array_image[j, i-1]+array_image[j+1, i-1]+array_image[j+1, i])
            k[6] = (-3)*(array_image[j-1, i-1]+array_image[j-1, i]+array_image[j-1, i+1]+array_image[j, i-1]+array_image[j, i+1])+5*(array_image[j+1, i-1]+array_image[j+1, i]+array_image[j+1, i+1])
            k[7] = (-3)*(array_image[j-1, i-1]+array_image[j-1, i]+array_image[j-1, i+1]+array_image[j, i-1]+array_image[j+1, i-1])+5*(array_image[j, i+1]+array_image[j+1, i]+array_image[j+1, i+1])
            gradient = np.amax(k)
            if(gradient < threshold):
                edge_image.putpixel((i-1, j-1), 255)
            else:
                edge_image.putpixel((i-1, j-1), 0)
    return edge_image

# Robinson compass operator
def Robinson_operator(image, threshold):
    array_image = np.array(image, dtype=int)
    array_image = np.pad(array_image, ((1, 1), (1, 1)), 'edge')
    edge_image = image.copy()
    for j in range(1, array_image.shape[0]-1):
        for i in range(1, array_image.shape[1]-1):
            r = np.zeros(8, dtype=int)
            r[0] = (array_image[j-1, i+1]+2*array_image[j, i+1]+array_image[j+1, i+1])-(array_image[j-1, i-1]+2*array_image[j, i-1]+array_image[j+1, i-1])
            r[1] = (array_image[j-1, i]+2*array_image[j-1, i+1]+array_image[j, i+1])-(array_image[j, i-1]+2*array_image[j+1, i-1]+array_image[j+1, i])
            r[2] = (array_image[j-1, i-1]+2*array_image[j-1, i]+array_image[j-1, i+1])-(array_image[j+1, i-1]+2*array_image[j+1, i]+array_image[j+1, i+1])
            r[3] = (array_image[j, i-1]+2*array_image[j-1, i-1]+array_image[j-1, i])-(array_image[j+1, i]+2*array_image[j+1, i+1]+array_image[j, i+1])
            r[4] = (array_image[j-1, i-1]+2*array_image[j, i-1]+array_image[j+1, i-1])-(array_image[j-1, i+1]+2*array_image[j, i+1]+array_image[j+1, i+1])
            r[5] = (array_image[j, i-1]+2*array_image[j+1, i-1]+array_image[j+1, i])-(array_image[j-1, i]+2*array_image[j-1, i+1]+array_image[j, i+1])
            r[6] = (array_image[j+1, i-1]+2*array_image[j+1, i]+array_image[j+1, i+1])-(array_image[j-1, i-1]+2*array_image[j-1, i]+array_image[j-1, i+1])
            r[7] = (array_image[j+1, i]+2*array_image[j+1, i+1]+array_image[j, i+1])-(array_image[j, i-1]+2*array_image[j-1, i-1]+array_image[j-1, i])
            gradient = np.amax(r)
            if(gradient < threshold):
                edge_image.putpixel((i-1, j-1), 255)
            else:
                edge_image.putpixel((i-1, j-1), 0)
    return edge_image

# Nevatia-Babu 5×5 operator
def NB_5x5_operator(image, threshold):
    array_image = np.array(image, dtype=int)
    array_image = np.pad(array_image, ((2, 2), (2, 2)), 'edge')
    edge_image = image.copy()
    for j in range(2, array_image.shape[0]-2):
        for i in range(2, array_image.shape[1]-2):
            n = np.zeros(6, dtype=int)
            NB_mask = np.array( [[[ 100,  100,  100,  100,  100],
                                  [ 100,  100,  100,  100,  100],
                                  [   0,    0,    0,    0,    0],
                                  [-100, -100, -100, -100, -100],
                                  [-100, -100, -100, -100, -100]],
                                
                                 [[ 100,  100,  100,  100,  100],
                                  [ 100,  100,  100,   78,  -32],
                                  [ 100,   92,    0,  -92, -100],
                                  [  32,  -78, -100, -100, -100],
                                  [-100, -100, -100, -100, -100]],
                                
                                 [[ 100,  100,  100,   32, -100],
                                  [ 100,  100,   92,  -78, -100],
                                  [ 100,  100,    0, -100, -100],
                                  [ 100,   78,  -92, -100, -100],
                                  [ 100,  -32, -100, -100, -100]],
                                
                                 [[-100, -100,    0,  100,  100],
                                  [-100, -100,    0,  100,  100],
                                  [-100, -100,    0,  100,  100],
                                  [-100, -100,    0,  100,  100],
                                  [-100, -100,    0,  100,  100]],

                                 [[-100,   32,  100,  100,  100],
                                  [-100,  -78,   92,  100,  100],
                                  [-100, -100,    0,  100,  100],
                                  [-100, -100,  -92,   78,  100],
                                  [-100, -100, -100,  -32,  100]],
                                
                                 [[ 100,  100,  100,  100,  100],
                                  [ -32,   78,  100,  100,  100],
                                  [-100,  -92,    0,   92,  100],
                                  [-100, -100, -100,  -78,   32],
                                  [-100, -100, -100, -100, -100]]] , dtype=int)
            for k in range(6):
                for b in range(5):
                    for a in range(5):
                        n[k] += NB_mask[k, b, a]*array_image[j+b-2, i+a-2]
            gradient = np.amax(n)
            if(gradient < threshold):
                edge_image.putpixel((i-2, j-2), 255)
            else:
                edge_image.putpixel((i-2, j-2), 0)
    return edge_image

Roberts_30 = Roberts_operator(im_raw, int(30))
Roberts_30.save("lena_Roberts_30.bmp")
Prewitt_24 = Prewitt_operator(im_raw, int(24))
Prewitt_24.save("lena_Prewitt_24.bmp")
Sobel_38 = Sobel_operator(im_raw, int(38))
Sobel_38.save("lena_Sobel_38.bmp")
FChen_30 = FC_operator(im_raw, int(30))
FChen_30.save("lena_FChen_30.bmp")
Kirsch_135 = Kirsch_operator(im_raw, int(135))
Kirsch_135.save("lena_Kirsch_135.bmp")
Robinson_43 = Robinson_operator(im_raw, int(43))
Robinson_43.save("lena_Robinson_43.bmp")
NB_5x5_12500 = NB_5x5_operator(im_raw, int(12500))
NB_5x5_12500.save("lena_NB_5x5_12500.bmp")
