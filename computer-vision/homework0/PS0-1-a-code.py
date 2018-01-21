"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-1-a'
output_extension = 'jpg'
im_names = ['TIB_8134.JPG', '4.2.04.tiff']

# load images
img = cv2.imread(im_names[0])
img2 = cv2.imread(im_names[1])


# print images and close on key press
cv2.imshow(im_names[0], img)
cv2.imshow(im_names[1], img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(question_name+'-1'+'.'+output_extension, img)
cv2.imwrite(question_name+'-2'+'.'+output_extension, img2)
