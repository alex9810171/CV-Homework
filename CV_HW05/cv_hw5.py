import numpy as np
from PIL import Image

im_raw = Image.open("lena.bmp")
kernel_octogonal = np.array([[0  , 255, 255, 255, 0  ],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [0  , 255, 255, 255, 0  ]], dtype=int)

# gray-scale dilation
def gray_scale_dilation(image, kernel, iterations):
    image_dilated = image.copy()
    kernel_x = kernel.shape[1]
    kernel_y = kernel.shape[0]
    for time in range(iterations):
        for j in range(image.size[1]):
            for i in range(image.size[0]):
                if(image.getpixel((i, j)) > 0
                 and i >= kernel_x//2 and i < (image.size[0]-(kernel_x//2))
                 and j >= kernel_y//2 and j < (image.size[1]-(kernel_y//2))):
                    max = 0
                    for l in range(j-(kernel_y//2), j+(kernel_y//2)+1, 1):
                        for k in range(i-(kernel_x//2), i+(kernel_x//2)+1, 1):
                            if(kernel[l-(j-(kernel_y//2)), k-(i-(kernel_x//2))].item() == 255
                             and image.getpixel((k, l)) > max):
                                max = image.getpixel((k, l))
                    image_dilated.putpixel((i, j), max)
    return image_dilated

# gray-scale erosion
def gray_scale_erosion(image, kernel, iterations):
    image_eroded = image.copy()
    kernel_x = kernel.shape[1]
    kernel_y = kernel.shape[0]
    for time in range(iterations):
        for j in range(image.size[1]):
            for i in range(image.size[0]):
                if(image.getpixel((i, j)) > 0
                 and i >= kernel_x//2 and i < (image.size[0]-(kernel_x//2))
                 and j >= kernel_y//2 and j < (image.size[1]-(kernel_y//2))):
                    min = 256
                    for l in range(j-(kernel_y//2), j+(kernel_y//2)+1, 1):
                        for k in range(i-(kernel_x//2), i+(kernel_x//2)+1, 1):
                            if(kernel[l-(j-(kernel_y//2)), k-(i-(kernel_x//2))].item() == 255
                             and image.getpixel((k, l)) < min):
                                min = image.getpixel((k, l))
                    image_eroded.putpixel((i, j), min)
    return image_eroded

gray_scale_dilation(im_raw, kernel_octogonal, 1).save("lena_dilated.bmp")
gray_scale_erosion(im_raw, kernel_octogonal, 1).save("lena_eroded.bmp")
gray_scale_dilation(gray_scale_erosion(im_raw, kernel_octogonal, 1), kernel_octogonal, 1).save("lena_opened.bmp")
gray_scale_erosion(gray_scale_dilation(im_raw, kernel_octogonal, 1), kernel_octogonal, 1).save("lena_closed.bmp")
