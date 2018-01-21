"""
Load and print the two sample images
"""

import cv2
import numpy as np
question_name = 'PS0-3-a-'
output_extension = 'jpg'
im_names = ['TIB_8134.JPG', '4.2.04.tiff']

# load images
img1 = cv2.imread(im_names[0])
img2 = cv2.imread(im_names[1])

# turn it monochrome
img1 = img1[:, :, 2]
img2 = img2[:, :, 2]

# computer coordinates of the tile to copy
# we assume that both image are larger that 100*100px
tile_origin1_x = int((img1.shape[0] - 100) * 0.5)
tile_origin1_y = int((img1.shape[1] - 100) * 0.5)
tile_origin2_x = int((img2.shape[0] - 100) * 0.5)
tile_origin2_y = int((img2.shape[1] - 100) * 0.5)

img2[tile_origin2_x:tile_origin2_x+100, tile_origin2_y:tile_origin2_y+100] = \
    img1[tile_origin1_x:tile_origin1_x+100, tile_origin1_y:tile_origin1_y+100]

# print images and close on key press
cv2.imshow(im_names[0]+'_tile', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(question_name+'.'+output_extension, img2)
