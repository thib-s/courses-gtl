"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-4-d-'
output_extension = 'jpg'
im_names = ['PS0-2-b-M1g.jpg']

# load images
img1 = cv2.imread(im_names[0])

kernel = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]])

img1 = cv2.filter2D(img1, -1, kernel, borderType=cv2.BORDER_CONSTANT)

# print images and close on key press
cv2.imshow(im_names[0]+'_tile', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(question_name+'.'+output_extension, img1)

