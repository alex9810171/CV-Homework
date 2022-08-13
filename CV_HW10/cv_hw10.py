import numpy as np
from PIL import Image

# input image
im_raw = Image.open("lena.bmp")

# Zero-Crossing Edge Image
def get_zero_crossing_edge_image(image, threshold, mask, norm):
    array_image = np.array(image, dtype=int)                            # image to array, then padding
    array_image = np.pad(array_image, ((mask.shape[0]//2, mask.shape[0]//2), (mask.shape[1]//2, mask.shape[1]//2)), 'edge')
    edge_array = np.zeros((image.size[0], image.size[1]), dtype=int)    # create -1/0/1 array
    for j in range(mask.shape[0]//2, array_image.shape[0]-mask.shape[0]//2):    # calculate gradient for each pixel in boundary
        for i in range(mask.shape[1]//2, array_image.shape[1]-mask.shape[1]//2):
            gradient = 0
            for b in range(mask.shape[0]):
                for a in range(mask.shape[1]):
                    gradient += mask[b, a]*array_image[j+b-mask.shape[0]//2, i+a-mask.shape[1]//2]
            gradient *= norm                                            # normalize factor
            if(gradient >= threshold):                                  # complete -1/0/1 array
                edge_array[j-mask.shape[0]//2, i-mask.shape[1]//2] = 1
            elif(gradient <= -threshold):
                edge_array[j-mask.shape[0]//2, i-mask.shape[1]//2] = -1
            else:
                edge_array[j-mask.shape[0]//2, i-mask.shape[1]//2] = 0
    edge_array = np.pad(edge_array, ((1, 1), (1, 1)), 'edge')
    edge_image = image.copy()                                           # create final image
    for j in range(1, edge_array.shape[0]-1):
        for i in range(1, edge_array.shape[1]-1):
            if(edge_array[j, i] == 1):                                  # check if there exists '-1' for each '1'
                detected = False                                        # pixel, if exists -> pixel = 0
                for b in range(3):                                      # not exists -> pixel = 255
                    for a in range(3):
                        if(edge_array[j+b-1, i+a-1] == -1):             # complete final image
                            edge_image.putpixel((i-1, j-1), 0)
                            detected = True
                        if(detected == True):
                            break
                    if(detected == True):
                        break
                if(detected == False):
                    edge_image.putpixel((i-1, j-1), 255)
            else:
                edge_image.putpixel((i-1, j-1), 255)
    return edge_image

Laplacian_v1_mask = np.array([[  0,  1,  0],
                              [  1, -4,  1],
                              [  0,  1,  0]], dtype=int)
Laplacian_v1 = get_zero_crossing_edge_image(im_raw, int(15), Laplacian_v1_mask, int(1))
Laplacian_v1.save("lena_Laplacian_v1.bmp")

Laplacian_v2_mask = np.array([[  1,  1,  1],
                              [  1, -8,  1],
                              [  1,  1,  1]], dtype=int)
Laplacian_v2 = get_zero_crossing_edge_image(im_raw, int(15), Laplacian_v2_mask, float(1/3))
Laplacian_v2.save("lena_Laplacian_v2.bmp")

Laplacian_min_var_mask = np.array([[  2, -1,  2],
                                   [ -1, -4, -1],
                                   [  2, -1,  2]], dtype=int)
Laplacian_min_var = get_zero_crossing_edge_image(im_raw, int(20), Laplacian_min_var_mask, float(1/3))
Laplacian_min_var.save("lena_Laplacian_min_var.bmp")

Laplacian_Gaussian_mask = np.array([[   0,   0,   0,  -1,  -1,  -2,  -1,  -1,   0,   0,   0],
		                            [   0,   0,  -2,  -4,  -8,  -9,  -8,  -4,  -2,   0,   0],
		                            [   0,  -2,  -7, -15, -22, -23, -22, -15,  -7,  -2,   0],
		                            [  -1,  -4, -15, -24, -14,  -1, -14, -24, -15,  -4,  -1],
		                            [  -1,  -8, -22, -14,  52, 103,  52, -14, -22,  -8,  -1],
		                            [  -2,  -9, -23,  -1, 103, 178, 103,  -1, -23,  -9,  -2],
		                            [  -1,  -8, -22, -14,  52, 103,  52, -14, -22,  -8,  -1],
	                            	[  -1,  -4, -15, -24, -14,  -1, -14, -24, -15,  -4,  -1],
	                            	[   0,  -2,  -7, -15, -22, -23, -22, -15,  -7,  -2,   0],
		                            [   0,   0,  -2,  -4,  -8,  -9,  -8,  -4,  -2,   0,   0],
		                            [   0,   0,   0,  -1,  -1,  -2,  -1,  -1,   0,   0,   0]], dtype=int)
Laplacian_Gaussian = get_zero_crossing_edge_image(im_raw, int(3000), Laplacian_Gaussian_mask, int(1))
Laplacian_Gaussian.save("lena_Laplacian_Gaussian.bmp")

Difference_Gaussian_mask = np.array([[  -1,  -3,  -4,  -6,  -7,  -8,  -7,  -6,  -4,  -3,  -1],
		                             [  -3,  -5,  -8, -11, -13, -13, -13, -11,  -8,  -5,  -3],
		                             [  -4,  -8, -12, -16, -17, -17, -17, -16, -12,  -8,  -4],
		                             [  -6, -11, -16, -16,   0,  15,   0, -16, -16, -11,  -6],
		                             [  -7, -13, -17,   0,  85, 160,  85,   0, -17, -13,  -7],
		                             [  -8, -13, -17,  15, 160, 283, 160,  15, -17, -13,  -8],
		                             [  -7, -13, -17,   0,  85, 160,  85,   0, -17, -13,  -7],
		                             [  -6, -11, -16, -16,   0,  15,   0, -16, -16, -11,  -6],
	                            	 [  -4,  -8, -12, -16, -17, -17, -17, -16, -12,  -8,  -4],
	                            	 [  -3,  -5,  -8, -11, -13, -13, -13, -11,  -8,  -5,  -3],
	                            	 [  -1,  -3,  -4,  -6,  -7,  -8,  -7,  -6,  -4,  -3,  -1]], dtype=int)
Difference_Gaussian = get_zero_crossing_edge_image(im_raw, int(1), Difference_Gaussian_mask, int(1))
Difference_Gaussian.save("lena_Difference_Gaussian.bmp")
