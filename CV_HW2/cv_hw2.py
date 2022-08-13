import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# input image
im_raw = Image.open("lena.bmp")

# (a) a binary image (threshold at 128)
im_binary = im_raw.copy()
for i in range(im_binary.size[0]):
    for j in range(im_binary.size[1]):
        if(im_binary.getpixel((i, j)) < 128):
            im_binary.putpixel((i, j), 0)
        else:
            im_binary.putpixel((i, j), 255)
im_binary.save("lena_binary.bmp")

# (b) a histogram
histogram_array = np.zeros(255).astype(int)
for i in range(im_raw.size[0]):
    for j in range(im_raw.size[1]):
        histogram_array[im_raw.getpixel((i, j))] += 1
plt_x = np.arange(len(histogram_array))
plt.bar(plt_x, histogram_array, width=1)
plt.xlim(0, 255)
plt.show()

# (c) connected components (regions with + at centroid, bounding box)
# pro-process image to array
im_bounding_box = im_binary.copy()
im_connected = np.array(im_bounding_box).astype(int)

# iterative algorithm to get component
# initialize each pixel to a unique label
label_count = 1
for i in range(im_connected.shape[1]):
    for j in range(im_connected.shape[0]):
        if(im_connected[i, j] == 255):
            im_connected[i, j] = label_count
            label_count += 1

change = True
while change == True:
    # top-down pass
    change = False
    for i in range(im_connected.shape[1]):
        for j in range(im_connected.shape[0]):
            # find a component
            if(im_connected[i, j] > 0):
                # first row
                if(j == 0):
                    # neighbor is a component
                    if(i > 0 and im_connected[i-1, j] > 0 and im_connected[i-1, j] < im_connected[i, j]):
                        im_connected[i, j] = im_connected[i-1, j]
                        change = True
                # second row or below
                else:
                    # neighbor exists at left and top
                    if(i > 0 and im_connected[i-1, j] > 0 and im_connected[i, j-1] > 0):
                        # choose a smaller component to override
                        if(im_connected[i-1, j] <= im_connected[i, j-1] and im_connected[i-1, j] < im_connected[i, j]):
                            im_connected[i, j] = im_connected[i-1, j]
                            change = True
                        elif(im_connected[i-1, j] > im_connected[i, j-1] and im_connected[i, j-1] < im_connected[i, j]):
                            im_connected[i, j] = im_connected[i, j-1]
                            change = True
                    # neighbor exists at left
                    elif(i > 0 and im_connected[i-1, j] > 0 and im_connected[i-1, j] < im_connected[i, j]):
                        im_connected[i, j] = im_connected[i-1, j]
                        change = True
                    # neighbor exists at top
                    elif(im_connected[i, j-1] > 0 and im_connected[i, j-1] < im_connected[i, j]):
                        im_connected[i, j] = im_connected[i, j-1]
                        change = True
    # bottom-up pass
    for i in range(im_connected.shape[1]-1, -1, -1):
        for j in range(im_connected.shape[0]-1, -1, -1):
            # find a component
            if(im_connected[i, j] > 0):
                # bottom row
                if(j == im_connected.shape[0]-1):
                    # neighbor is a component
                    if(i < im_connected.shape[1]-1 and im_connected[i+1, j] > 0 and im_connected[i+1, j] < im_connected[i, j]):
                        im_connected[i, j] = im_connected[i+1, j]
                        change = True
                # upon bottom row
                else:
                    # neighbor exists at right and bottom
                    if(i < im_connected.shape[1]-1 and im_connected[i+1, j] > 0 and im_connected[i, j+1] > 0):
                        # choose a smaller component to override
                        if(im_connected[i+1, j] <= im_connected[i, j+1] and im_connected[i+1, j] < im_connected[i, j]):
                            im_connected[i, j] = im_connected[i+1, j]
                            change = True
                        elif(im_connected[i+1, j] > im_connected[i, j+1] and im_connected[i, j+1] < im_connected[i, j]):
                            im_connected[i, j] = im_connected[i, j+1]
                            change = True
                    # neighbor exists at right
                    elif(i < im_connected.shape[1]-1 and im_connected[i+1, j] > 0 and im_connected[i+1, j] < im_connected[i, j]):
                        im_connected[i, j] = im_connected[i+1, j]
                        change = True
                    # neighbor exists at bottom
                    elif(im_connected[i, j+1] > 0 and im_connected[i, j+1] < im_connected[i, j]):
                        im_connected[i, j] = im_connected[i, j+1]
                        change = True

# create dictionary
dic = {}
for i in range(im_connected.shape[1]):
    for j in range(im_connected.shape[0]):
        if(im_connected[i, j] > 0):
            if(im_connected[i, j] not in dic):
                dic[im_connected[i, j]] = []
            dic[im_connected[i, j]].append([j, i])

# calculate bounding box and centroid
component_centroid = []
component_bounding_box = []
for i in range(1, label_count, 1):
    if(i in dic and len(dic[i]) >= 500):
        centroid_x = 0
        centroid_y = 0
        bounding_box = [512, 512, -1, -1]
        for j in range(len(dic[i])):
            x = dic[i][j][0]
            y = dic[i][j][1]
            centroid_x += x
            centroid_y += y
            if(x < bounding_box[0]):
                bounding_box[0] = x
            if(x > bounding_box[2]):
                bounding_box[2] = x
            if(y < bounding_box[1]):
                bounding_box[1] = y
            if(y > bounding_box[3]):
                bounding_box[3] = y
        component_centroid.append([centroid_x // len(dic[i]), centroid_y // len(dic[i])])
        component_bounding_box.append(bounding_box)

# draw bounding box
im_bounding_box = im_bounding_box.convert('RGB')
draw = ImageDraw.Draw(im_bounding_box)
for i in range(len(component_bounding_box)):
    bound = component_bounding_box[i]
    draw.rectangle([bound[0], bound[1], bound[2], bound[3]], outline='blue', width=3)
    x = component_centroid[i][0]
    y = component_centroid[i][1]
    draw.line([x-10, y, x+10, y], fill='red', width=2)
    draw.line([x, y-10, x, y+10], fill='red', width=2)
del draw
im_bounding_box.save("lena_bounding_box.bmp")

''' classic algorithm
equivalence_table = []

# first top-down pass
component_count = 1
for i in range(im_connected.shape[0]):
    for j in range(im_connected.shape[1]):
        # find a component
        if(im_connected[i, j] > 0):
            # first row
            if(j == 0):
                # neighbor is a component
                if(im_connected[i-1, j] > 0 and i > 0):
                    im_connected[i, j] = im_connected[i-1, j]
                # no neighbor
                else:
                    im_connected[i, j] = component_count
                    component_count += 1
            
            # second row or below
            else:
                # neighbor exists at left and top
                if(im_connected[i-1, j] > 0 and im_connected[i, j-1] > 0 and i > 0):
                    # choose a smaller component to override
                    if(im_connected[i-1, j] < im_connected[i, j-1]):
                        im_connected[i, j] = im_connected[i-1, j]
                        equivalence_table.append([im_connected[i-1, j], im_connected[i, j-1]])
                    else:
                        im_connected[i, j] = im_connected[i, j-1]
                        equivalence_table.append([im_connected[i, j-1], im_connected[i-1, j]])
                # neighbor exists at left
                elif(im_connected[i-1, j] > 0 and im_connected[i, j-1] == 0 and i > 0):
                    im_connected[i, j] = im_connected[i-1, j]
                # neighbor exists at top
                elif(im_connected[i, j-1] > 0):
                    im_connected[i, j] = im_connected[i, j-1]
                # no neighbor
                else:
                    im_connected[i, j] = component_count
                    component_count += 1

# second top-down pass
def find_smallest_component(connected_id):
    for i in range(len(equivalence_table)):
        equivalence = equivalence_table[i]
        if(connected_id == equivalence[1]):
            return find_smallest_component(equivalence[0])
    return connected_id
            
for i in range(im_connected.shape[0]):
    for j in range(im_connected.shape[1]):
        # find a component and replace it
        if(im_connected[i, j] > 0):
            im_connected[i, j] = find_smallest_component(im_connected[i, j])
print(equivalence_table)

==================
for i in range(im_connected.shape[0]):
    for j in range(im_connected.shape[1]):
        if(im_connected[i, j] == 255):
            im_connected[i, j] = -1

# variable
component_size = []
component_centroid = []
component_bounding_box = []
size = 0
color = 1
centroid_x = 0
print(type(centroid_x))
centroid_y = 0
print(type(centroid_y))
bounding_box = [512, 512, -1, -1]

# flood_fill with DFS algorithm
def flood_fill(x, y, fill_color, overrided_color):
    global im_connected
    global size
    global centroid_x
    global centroid_y
    global bounding_box
    im_connected[x, y] = fill_color
    size += 1
    centroid_x += x
    centroid_y += y
    if(x <= bounding_box[0] and y <= bounding_box[1]):
        bounding_box[0] = x
        bounding_box[1] = y
    if(x >= bounding_box[2] and y >= bounding_box[3]):
        bounding_box[2] = x
        bounding_box[3] = y

    # iteration
    if(x > 0 and im_connected[x-1, y] == overrided_color):
        flood_fill(x-1, y, fill_color, overrided_color)
    if(y > 0 and im_connected[x, y-1] == overrided_color):
        flood_fill(x, y-1, fill_color, overrided_color)
    if(x < im_connected.shape[0]-1 and im_connected[x+1, y] == overrided_color):
        flood_fill(x+1, y, fill_color, overrided_color)
    if(y < im_connected.shape[1]-1 and im_connected[x, y+1] == overrided_color):
        flood_fill(x, y+1, fill_color, overrided_color)

    for i in range(im_connected.shape[0]):
    for j in range(im_connected.shape[1]):
        if(im_connected[i, j] == -1):
            flood_fill(i, j, color, -1)
            print("hello world")
            # if size too small, delete component using 0
            if(size < 500):
                flood_fill(i, j, 0, color)
            # otherwise, record it
            else:
                component_size.append(size)
                component_centroid.append([centroid_x / size, centroid_y / size])
                component_bounding_box.append(bounding_box)
                color += 1
            # initialize size and bounding box
            size = 0
            centroid_x = 0
            centroid_y = 0
            bounding_box[0] = 512
            bounding_box[1] = 512
            bounding_box[2] = -1
            bounding_box[3] = -1
'''