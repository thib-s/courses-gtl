"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-5-b-'
output_extension = 'jpg'
im_names = ['PS0-1-a-1.jpg']

# load images
img1 = cv2.imread(im_names[0])

gaussian_img = np.random.normal(0, 10, (img1.shape[0], img1.shape[1])).astype(int)
img1[:, :, 0] = cv2.add(img1[:, :, 0].astype(int), gaussian_img)

# print images and close on key press
cv2.imshow(im_names[0]+'_tile', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(question_name+'.'+output_extension, img1)
