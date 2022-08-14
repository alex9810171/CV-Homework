import math
import random
import numpy as np
from PIL import Image

# input image
im_raw = Image.open("lena.bmp")

# Gaussian noise 
def get_gaussian_noise_image(ori_image, amplitude):
    gaussian_noise_image = ori_image.copy()
    for j in range(ori_image.size[1]):
        for i in range(ori_image.size[0]):
            pixel = int(ori_image.getpixel((i, j)) + amplitude*random.gauss(0, 1))
            if(pixel > 255):
                pixel = 255
            elif(pixel < 0):
                pixel = 0
            gaussian_noise_image.putpixel((i, j), pixel)
    return gaussian_noise_image

# salt-and-pepper noise 
def get_salt_and_pepper_noise_image(ori_image, threshold):
    salt_and_pepper_noise_image = ori_image.copy()
    for j in range(ori_image.size[1]):
        for i in range(ori_image.size[0]):
            randomValue = random.uniform(0, 1)
            if(randomValue < threshold):
                salt_and_pepper_noise_image.putpixel((i, j), 0)
            elif(randomValue > 1-threshold):
                salt_and_pepper_noise_image.putpixel((i, j), 255)
    return salt_and_pepper_noise_image

# box filter on noisy images
def get_box_filter_image(noisy_image, size):
    array_noisy_image = np.array(noisy_image)
    array_pad_noisy_image = np.pad(array_noisy_image, ((size[0]//2, size[0]//2), (size[1]//2, size[1]//2)), 'edge')
    array_box_filter_image = array_noisy_image.copy()
    for j in range(size[0]//2, array_pad_noisy_image.shape[0]-size[0]//2):
        for i in range(size[1]//2, array_pad_noisy_image.shape[1]-size[1]//2):
            total = 0
            for l in range(j-size[0]//2, j+size[0]//2+1):
                for k in range(i-size[1]//2, i+size[1]//2+1):
                    total += array_pad_noisy_image[l, k].item()
            total //= size[0]*size[1]
            array_box_filter_image[j-size[0]//2, i-size[1]//2] = total
    return Image.fromarray(array_box_filter_image.astype(np.uint8))

# median filter on noisy images
def get_median_filter_image(noisy_image, size):
    array_noisy_image = np.array(noisy_image)
    array_pad_noisy_image = np.pad(array_noisy_image, ((size[0]//2, size[0]//2), (size[1]//2, size[1]//2)), 'edge')
    array_median_filter_image = array_noisy_image.copy()
    for j in range(size[0]//2, array_pad_noisy_image.shape[0]-size[0]//2):
        for i in range(size[1]//2, array_pad_noisy_image.shape[1]-size[1]//2):
            array_median_filter = np.zeros((size[0], size[1])).astype(int)
            for l in range(j-size[0]//2, j+size[0]//2+1):
                for k in range(i-size[1]//2, i+size[1]//2+1):
                    array_median_filter[l-(j-size[0]//2), k-(i-size[1]//2)] = array_pad_noisy_image[l, k].item()
            array_median_filter_image[j-size[0]//2, i-size[1]//2] = np.median(array_median_filter)
    return Image.fromarray(array_median_filter_image.astype(np.uint8))

# 3-5-5-5-3 kernel
kernel_octogonal = np.array([[0  , 255, 255, 255, 0  ],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [0  , 255, 255, 255, 0  ]], dtype=int)

# gray-scale dialtion
def gray_scale_dilation(image, kernel, iterations):
    image_dilated = image.copy()
    for time in range(iterations):
        for j in range(kernel.shape[0]//2, image.size[1]-kernel.shape[0]//2):
            for i in range(kernel.shape[1]//2, image.size[0]-kernel.shape[1]//2):
                local_max = 0
                for l in range(j-kernel.shape[0]//2, j+kernel.shape[0]//2+1):
                    for k in range(i-kernel.shape[1]//2, i+kernel.shape[1]//2+1):
                        if(kernel[l-(j-kernel.shape[0]//2), k-(i-kernel.shape[1]//2)] == 255 and int(image.getpixel((k, l))) > local_max):
                            local_max = int(image.getpixel((k, l)))
                image_dilated.putpixel((i, j), local_max)
    return image_dilated

def gray_scale_erosion(image, kernel, iterations):
    image_eroded = image.copy()
    for time in range(iterations):
        for j in range(kernel.shape[0]//2, image.size[1]-kernel.shape[0]//2):
            for i in range(kernel.shape[1]//2, image.size[0]-kernel.shape[1]//2):
                local_min = 255
                for l in range(j-kernel.shape[0]//2, j+kernel.shape[0]//2+1):
                    for k in range(i-kernel.shape[1]//2, i+kernel.shape[1]//2+1):
                        if(kernel[l-(j-kernel.shape[0]//2), k-(i-kernel.shape[1]//2)] == 255 and int(image.getpixel((k, l))) < local_min):
                            local_min = int(image.getpixel((k, l)))
                image_eroded.putpixel((i, j), local_min)
    return image_eroded

# gray-scale opening
def gray_scale_opening(image, kernel, iterations):
    return gray_scale_dilation(gray_scale_erosion(image, kernel, iterations), kernel, iterations)

# gray-scale closing
def gray_scale_closing(image, kernel, iterations):
    return gray_scale_erosion(gray_scale_dilation(image, kernel, iterations), kernel, iterations)

# count SNR
def count_SNR(ori_image, noise_image):
    norm_ori_image = np.zeros((ori_image.size[1], ori_image.size[0])).astype(float)
    norm_noise_image = np.zeros((noise_image.size[1], noise_image.size[0])).astype(float)
    for j in range(norm_ori_image.shape[0]):
        for i in range(norm_ori_image.shape[1]):
            norm_ori_image[j, i] = ori_image.getpixel((i, j))/255
    for j in range(norm_noise_image.shape[0]):
        for i in range(norm_noise_image.shape[1]):
            norm_noise_image[j, i] = noise_image.getpixel((i, j))/255

    total_illuminance_of_ori = 0
    for j in range(norm_ori_image.shape[0]):
        for i in range(norm_ori_image.shape[1]):
            total_illuminance_of_ori += norm_ori_image[j, i]
    mean_of_ori = total_illuminance_of_ori / (norm_ori_image.shape[1]*norm_ori_image.shape[0])

    total_variance_of_ori = 0
    for j in range(norm_ori_image.shape[0]):
        for i in range(norm_ori_image.shape[1]):
            total_variance_of_ori += (norm_ori_image[j, i]-mean_of_ori)**2
    variance_of_signal = total_variance_of_ori / (norm_ori_image.shape[1]*norm_ori_image.shape[0])

    total_noise = 0
    for j in range(norm_ori_image.shape[0]):
        for i in range(norm_ori_image.shape[1]):
            total_noise += (norm_noise_image[j, i]-norm_ori_image[j, i])
    mean_of_noise = total_noise / (norm_ori_image.shape[1]*norm_ori_image.shape[0])

    total_variance_of_noise = 0
    for j in range(norm_ori_image.shape[1]):
        for i in range(norm_ori_image.shape[0]):
            total_variance_of_noise += (norm_noise_image[j, i]-norm_ori_image[j, i]-mean_of_noise)**2
    variance_of_noise = total_variance_of_noise / (norm_ori_image.shape[1]*norm_ori_image.shape[0])

    return 20*math.log(variance_of_signal**0.5 / variance_of_noise**0.5, 10)

# main part
gaussian_noise_image_10 = get_gaussian_noise_image(im_raw, 10)
gaussian_noise_image_10.save("lena_gaussian_noise_image_10.bmp")
print("SNR of Gaussian noise image with amplitude 10 is: %f" %(count_SNR(im_raw, gaussian_noise_image_10)))
gaussian_noise_image_30 = get_gaussian_noise_image(im_raw, 30)
gaussian_noise_image_30.save("lena_gaussian_noise_image_30.bmp")
print("SNR of Gaussian noise image with amplitude 30 is: %f" %(count_SNR(im_raw, gaussian_noise_image_30)))

salt_and_pepper_noise_image_005 = get_salt_and_pepper_noise_image(im_raw, 0.05)
salt_and_pepper_noise_image_005.save("lena_salt_and_pepper_noise_image_005.bmp")
print("SNR of salt and pepper noise image with threshold 0.05 is: %f" %(count_SNR(im_raw, salt_and_pepper_noise_image_005)))
salt_and_pepper_noise_image_010 = get_salt_and_pepper_noise_image(im_raw, 0.10)
salt_and_pepper_noise_image_010.save("lena_salt_and_pepper_noise_image_010.bmp")
print("SNR of salt and pepper noise image with threshold 0.10 is: %f" %(count_SNR(im_raw, salt_and_pepper_noise_image_010)))

box_filter_3x3_gaussian_image_10 = get_box_filter_image(gaussian_noise_image_10, np.array([3, 3]))
box_filter_3x3_gaussian_image_10.save("lena_box_filter_3x3_gaussian_image_10.bmp")
print("SNR of box filter 3x3 on Gaussian noise image with amplitude 10 is: %f" %(count_SNR(im_raw, box_filter_3x3_gaussian_image_10)))
box_filter_3x3_gaussian_image_30 = get_box_filter_image(gaussian_noise_image_30, np.array([3, 3]))
box_filter_3x3_gaussian_image_30.save("lena_box_filter_3x3_gaussian_image_30.bmp")
print("SNR of box filter 3x3 on Gaussian noise image with amplitude 30 is: %f" %(count_SNR(im_raw, box_filter_3x3_gaussian_image_30)))
box_filter_3x3_salt_and_pepper_image_005 = get_box_filter_image(salt_and_pepper_noise_image_005, np.array([3, 3]))
box_filter_3x3_salt_and_pepper_image_005.save("lena_box_filter_3x3_salt_and_pepper_image_005.bmp")
print("SNR of box filter 3x3 on salt and pepper noise image with threshold 0.05 is: %f" %(count_SNR(im_raw, box_filter_3x3_salt_and_pepper_image_005)))
box_filter_3x3_salt_and_pepper_image_010 = get_box_filter_image(salt_and_pepper_noise_image_010, np.array([3, 3]))
box_filter_3x3_salt_and_pepper_image_010.save("lena_box_filter_3x3_salt_and_pepper_image_010.bmp")
print("SNR of box filter 3x3 on salt and pepper noise image with threshold 0.10 is: %f" %(count_SNR(im_raw, box_filter_3x3_salt_and_pepper_image_010)))
box_filter_5x5_gaussian_image_10 = get_box_filter_image(gaussian_noise_image_10, np.array([5, 5]))
box_filter_5x5_gaussian_image_10.save("lena_box_filter_5x5_gaussian_image_10.bmp")
print("SNR of box filter 5x5 on Gaussian noise image with amplitude 10 is: %f" %(count_SNR(im_raw, box_filter_5x5_gaussian_image_10)))
box_filter_5x5_gaussian_image_30 = get_box_filter_image(gaussian_noise_image_30, np.array([5, 5]))
box_filter_5x5_gaussian_image_30.save("lena_box_filter_5x5_gaussian_image_30.bmp")
print("SNR of box filter 5x5 on Gaussian noise image with amplitude 30 is: %f" %(count_SNR(im_raw, box_filter_5x5_gaussian_image_30)))
box_filter_5x5_salt_and_pepper_image_005 = get_box_filter_image(salt_and_pepper_noise_image_005, np.array([5, 5]))
box_filter_5x5_salt_and_pepper_image_005.save("lena_box_filter_5x5_salt_and_pepper_image_005.bmp")
print("SNR of box filter 5x5 on salt and pepper noise image with threshold 0.05 is: %f" %(count_SNR(im_raw, box_filter_5x5_salt_and_pepper_image_005)))
box_filter_5x5_salt_and_pepper_image_010 = get_box_filter_image(salt_and_pepper_noise_image_010, np.array([5, 5]))
box_filter_5x5_salt_and_pepper_image_010.save("lena_box_filter_5x5_salt_and_pepper_image_010.bmp")
print("SNR of box filter 5x5 on salt and pepper noise image with threshold 0.10 is: %f" %(count_SNR(im_raw, box_filter_5x5_salt_and_pepper_image_010)))

median_filter_3x3_gaussian_image_10 = get_median_filter_image(gaussian_noise_image_10, np.array([3, 3]))
median_filter_3x3_gaussian_image_10.save("lena_median_filter_3x3_gaussian_image_10.bmp")
print("SNR of median filter 3x3 on Gaussian noise image with amplitude 10 is: %f" %(count_SNR(im_raw, median_filter_3x3_gaussian_image_10)))
median_filter_3x3_gaussian_image_30 = get_median_filter_image(gaussian_noise_image_30, np.array([3, 3]))
median_filter_3x3_gaussian_image_30.save("lena_median_filter_3x3_gaussian_image_30.bmp")
print("SNR of median filter 3x3 on Gaussian noise image with amplitude 30 is: %f" %(count_SNR(im_raw, median_filter_3x3_gaussian_image_30)))
median_filter_3x3_salt_and_pepper_image_005 = get_median_filter_image(salt_and_pepper_noise_image_005, np.array([3, 3]))
median_filter_3x3_salt_and_pepper_image_005.save("lena_median_filter_3x3_salt_and_pepper_image_005.bmp")
print("SNR of median filter 3x3 on salt and pepper noise image with threshold 0.05 is: %f" %(count_SNR(im_raw, median_filter_3x3_salt_and_pepper_image_005)))
median_filter_3x3_salt_and_pepper_image_010 = get_median_filter_image(salt_and_pepper_noise_image_010, np.array([3, 3]))
median_filter_3x3_salt_and_pepper_image_010.save("lena_median_filter_3x3_salt_and_pepper_image_010.bmp")
print("SNR of median filter 3x3 on salt and pepper noise image with threshold 0.10 is: %f" %(count_SNR(im_raw, median_filter_3x3_salt_and_pepper_image_010)))
median_filter_5x5_gaussian_image_10 = get_median_filter_image(gaussian_noise_image_10, np.array([5, 5]))
median_filter_5x5_gaussian_image_10.save("lena_median_filter_5x5_gaussian_image_10.bmp")
print("SNR of median filter 5x5 on Gaussian noise image with amplitude 10 is: %f" %(count_SNR(im_raw, median_filter_5x5_gaussian_image_10)))
median_filter_5x5_gaussian_image_30 = get_median_filter_image(gaussian_noise_image_30, np.array([5, 5]))
median_filter_5x5_gaussian_image_30.save("lena_median_filter_5x5_gaussian_image_30.bmp")
print("SNR of median filter 5x5 on Gaussian noise image with amplitude 30 is: %f" %(count_SNR(im_raw, median_filter_5x5_gaussian_image_30)))
median_filter_5x5_salt_and_pepper_image_005 = get_median_filter_image(salt_and_pepper_noise_image_005, np.array([5, 5]))
median_filter_5x5_salt_and_pepper_image_005.save("lena_median_filter_5x5_salt_and_pepper_image_005.bmp")
print("SNR of median filter 5x5 on salt and pepper noise image with threshold 0.05 is: %f" %(count_SNR(im_raw, median_filter_5x5_salt_and_pepper_image_005)))
median_filter_5x5_salt_and_pepper_image_010 = get_median_filter_image(salt_and_pepper_noise_image_010, np.array([5, 5]))
median_filter_5x5_salt_and_pepper_image_010.save("lena_median_filter_5x5_salt_and_pepper_image_010.bmp")
print("SNR of median filter 5x5 on salt and pepper noise image with threshold 0.10 is: %f" %(count_SNR(im_raw, median_filter_5x5_salt_and_pepper_image_010)))

opening_and_closing_gaussian_image_10 = gray_scale_closing(gray_scale_opening(gaussian_noise_image_10, kernel_octogonal, 1), kernel_octogonal, 1)
opening_and_closing_gaussian_image_10.save("lena_opening_and_closing_gaussian_image_10.bmp")
print("SNR of opening and closing on Gaussian noise image with amplitude 10 is: %f" %(count_SNR(im_raw, opening_and_closing_gaussian_image_10)))
opening_and_closing_gaussian_image_30 = gray_scale_closing(gray_scale_opening(gaussian_noise_image_30, kernel_octogonal, 1), kernel_octogonal, 1)
opening_and_closing_gaussian_image_30.save("lena_opening_and_closing_gaussian_image_30.bmp")
print("SNR of opening and closing on Gaussian noise image with amplitude 30 is: %f" %(count_SNR(im_raw, opening_and_closing_gaussian_image_30)))
opening_and_closing_salt_and_pepper_image_005 = gray_scale_closing(gray_scale_opening(salt_and_pepper_noise_image_005, kernel_octogonal, 1), kernel_octogonal, 1)
opening_and_closing_salt_and_pepper_image_005.save("lena_opening_and_closing_salt_and_pepper_image_005.bmp")
print("SNR of opening and closing on salt and pepper noise image with threshold 0.05 is: %f" %(count_SNR(im_raw, opening_and_closing_salt_and_pepper_image_005)))
opening_and_closing_salt_and_pepper_image_010 = gray_scale_closing(gray_scale_opening(salt_and_pepper_noise_image_010, kernel_octogonal, 1), kernel_octogonal, 1)
opening_and_closing_salt_and_pepper_image_010.save("lena_opening_and_closing_salt_and_pepper_image_010.bmp")
print("SNR of opening and closing on salt and pepper noise image with threshold 0.10 is: %f" %(count_SNR(im_raw, opening_and_closing_salt_and_pepper_image_010)))
closing_and_opening_gaussian_image_10 = gray_scale_opening(gray_scale_closing(gaussian_noise_image_10, kernel_octogonal, 1), kernel_octogonal, 1)
closing_and_opening_gaussian_image_10.save("lena_closing_and_opening_gaussian_image_10.bmp")
print("SNR of closing and opening on Gaussian noise image with amplitude 10 is: %f" %(count_SNR(im_raw, closing_and_opening_gaussian_image_10)))
closing_and_opening_gaussian_image_30 = gray_scale_opening(gray_scale_closing(gaussian_noise_image_30, kernel_octogonal, 1), kernel_octogonal, 1)
closing_and_opening_gaussian_image_30.save("lena_closing_and_opening_gaussian_image_30.bmp")
print("SNR of closing and opening on Gaussian noise image with amplitude 30 is: %f" %(count_SNR(im_raw, closing_and_opening_gaussian_image_30)))
closing_and_opening_salt_and_pepper_image_005 = gray_scale_opening(gray_scale_closing(salt_and_pepper_noise_image_005, kernel_octogonal, 1), kernel_octogonal, 1)
closing_and_opening_salt_and_pepper_image_005.save("lena_closing_and_opening_salt_and_pepper_image_005.bmp")
print("SNR of closing and opening on salt and pepper noise image with threshold 0.05 is: %f" %(count_SNR(im_raw, closing_and_opening_salt_and_pepper_image_005)))
closing_and_opening_salt_and_pepper_image_010 = gray_scale_opening(gray_scale_closing(salt_and_pepper_noise_image_010, kernel_octogonal, 1), kernel_octogonal, 1)
closing_and_opening_salt_and_pepper_image_010.save("lena_closing_and_opening_salt_and_pepper_image_010.bmp")
print("SNR of closing and opening on salt and pepper noise image with threshold 0.10 is: %f" %(count_SNR(im_raw, closing_and_opening_salt_and_pepper_image_010)))
