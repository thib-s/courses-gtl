"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-2-a'
output_extension = 'jpg'
im_names = ['TIB_8134.JPG', '4.2.04.tiff']

# load images
img = cv2.imread(im_names[0])

# swap channels
blue_buf = img[:, :, 0]
img[:, :, 0] = img[:, :, 2]
img[:, :, 2] = blue_buf

# print images and close on key press
cv2.imshow(im_names[0]+'_swapped', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(question_name+'.'+output_extension, img)
