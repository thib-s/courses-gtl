"""
Load and print the two sample images
"""

import argparse
import cv2
import numpy as np
question_name = 'sobel-'
output_extension = 'jpg'

parser = argparse.ArgumentParser(description='Apply sobel filter on an image')
parser.add_argument('filepath', metavar='N', type=string,
                    help='the path to the image to process')

args = parser.parse_args()
im_names = [args.filepath]

# load images
img1 = cv2.imread(im_names[0])[ :, :, 2]

kernel_smooth = np.vectorize(lambda x: x*0.1)(np.array(
    [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
)
)
#img1 = cv2.GaussianBlur( src=img1, ksize=(27,27), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_REFLECT)
#img1 = cv2.filter2D(img1, -1, kernel_smooth, borderType=cv2.BORDER_CONSTANT)

kernel = np.vectorize(lambda x: x*0.125)(np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
))
kernel_h = np.vectorize(lambda x: x*0.125)(np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
))


img_x = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=3)#cv2.filter2D(img1, -1, kernel, borderType=cv2.BORDER_REFLECT)
img_y = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=3)#cv2.filter2D(img1, -1, kernel_h, borderType=cv2.BORDER_REFLECT)

vfunc = np.vectorize(lambda t: t ** 2)

img = np.vectorize(np.math.sqrt)(np.add(
        vfunc(img_x),
        vfunc(img_y)
    )
)
sigma = np.max(img)
img = np.vectorize(lambda x: int(x*255/sigma))(img)

# print images and close on key press
#cv2.imshow(im_names[0]+'_tile', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite(question_name+im_names[0]+'.'+output_extension, img)

