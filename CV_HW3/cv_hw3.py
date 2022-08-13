import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# input image
im_raw = Image.open("lena.bmp")

# (a) original image and its histogram
def draw_histogram(im_draw):
    histogram_array = np.zeros(256).astype(int)
    for i in range(im_draw.size[0]):
        for j in range(im_draw.size[1]):
            histogram_array[im_draw.getpixel((i, j))] += 1
    plt_x = np.arange(len(histogram_array))
    plt.bar(plt_x, histogram_array, width=1)
    plt.xlim(0, 255)
    plt.show()
    return histogram_array
draw_histogram(im_raw)

# (b) image with intensity divided by 3 and its histogram
im_low_intensity = im_raw.copy()
for i in range(im_low_intensity.size[0]):
    for j in range(im_low_intensity.size[1]):
            color = im_low_intensity.getpixel((i, j))//3
            im_low_intensity.putpixel((i, j), color)
im_low_intensity.save("lena_low_intensity.bmp")
histogram_array = draw_histogram(im_low_intensity)

# (c) image after applying histogram equalization to (b) and its histogram
im_he = im_low_intensity.copy()

# get pdf
pdf_array = np.zeros(256).astype(float)
for i in range(len(pdf_array)):
    mn = im_he.size[0]*im_he.size[1]
    pdf_array[i] = histogram_array[i] / mn

# get cdf
cdf_array = np.zeros(256).astype(float)
for i in range(len(cdf_array)):
    for j in range(i+1):
        cdf_array[i] += pdf_array[j]
    cdf_array[i] *= 255

# from intensity k to Sk
for i in range(im_he.size[0]):
    for j in range(im_he.size[1]):
        color = round(cdf_array[im_he.getpixel((i, j))])
        im_he.putpixel((i, j), color)

im_he.save("lena_histogram_equalization.bmp")
draw_histogram(im_he)
