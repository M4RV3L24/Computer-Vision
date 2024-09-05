from PIL import Image
import numpy as np
from functions import *

image_array, rows, cols = readImage('images/cat.jpg')

filter_array = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        filter_array[i, j] = 1/25
        
rdifs = filter_array.shape[0] // 2
cdifs = filter_array.shape[1] // 2

filtered_array = np.copy(image_array)
for row in range(rows):
    for col in range(cols // 2):
        rgb = np.array([0, 0, 0])
        for rdif in range(-rdifs, rdifs+1):
            for cdif in range(-cdifs, cdifs+1):
                if (row + rdif) >= rows or (row + rdif) < 0 or (col + cdif) >= cols or (col + cdif) < 0:
                    continue
                for idx in range(0, 3):
                    rgb[idx] += image_array[row + rdif, col + cdif, idx] * filter_array[rdif + rdifs, cdif + cdifs]
        filtered_array[row, col] = rgb
        
saveImage(filtered_array, 'images/filtered-cat-half.jpg')
combineImage('images/cat.jpg', 'images/filtered-cat-half.jpg', 'images/combined-cat-half.jpg', True)
        