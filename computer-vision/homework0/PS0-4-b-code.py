"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-4-b-'
output_extension = 'jpg'
im_names = ['PS0-2-b-M1g.jpg']

# load images
img1 = cv2.imread(im_names[0])

# flatten image
flat = np.ndarray.flatten(img1)

# compute values
mean_val = np.mean(flat)
std_dev_val = np.std(flat)

# display infos for debugging purpose
print("mean value of the image is :" + str(mean_val))
print("std dev of the image is :" + str(std_dev_val))
print("max value of the image is :" + str(max(flat)))
print("max value of the image is :" + str(min(flat)))

# remove the mean value
tmp_img = np.subtract(img1, np.multiply(np.ones(img1.shape), mean_val))
# divide by the sd_dev
tmp_img = np.divide(tmp_img, np.multiply(np.ones(tmp_img.shape), std_dev_val))
# multiply by 10
tmp_img = np.multiply(tmp_img, np.multiply(np.ones(tmp_img.shape), 10))
# add mean back in
img1 = cv2.add(img1, np.multiply(np.ones(img1.shape, img1.dtype), int(mean_val)))


# print images and close on key press
cv2.imshow(im_names[0]+'_tile', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(question_name+'.'+output_extension, img1)
