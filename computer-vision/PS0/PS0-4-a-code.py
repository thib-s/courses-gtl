"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-4-a-'
output_extension = 'jpg'
im_names = ['PS0-2-b-M1g.jpg']

# load images
img1 = cv2.imread(im_names[0])

# flatten image
flat = np.ndarray.flatten(img1)

# compute mean
print("mean value of the image is :" + str(np.mean(flat)))
print("std dev of the image is :" + str(np.std(flat)))
print("max value of the image is :" + str(max(flat)))
print("max value of the image is :" + str(min(flat)))
