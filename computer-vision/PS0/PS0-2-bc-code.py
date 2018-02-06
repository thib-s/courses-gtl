"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-2'
output_extension = 'jpg'
im_names = ['TIB_8134.JPG', '4.2.04.tiff']

# load images
img = cv2.imread(im_names[0])


# print images and close on key press
cv2.imshow(im_names[0]+'0', img[:, :, 0])  # blue channel
cv2.imshow(im_names[0]+'1', img[:, :, 1])  # green channel
cv2.imshow(im_names[0]+'2', img[:, :, 2])  # red channel
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(question_name+'-b-M1g'+'.'+output_extension, img[:, :, 1])
cv2.imwrite(question_name+'-c-M1r'+'.'+output_extension, img[:, :, 2])
