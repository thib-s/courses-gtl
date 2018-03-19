import cv2
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_3d_img(matches):
    # Set up grid and test data
    (imy, imx) = matches.shape
    x = range(imx)
    y = range(imy)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, matches)

    plt.show()


def display_img(img):
    pylab.imshow(img, cmap=pylab.gray())
    pylab.show()


def display_and_save(img, name):
    pylab.imshow(img, cmap=pylab.gray())
    pylab.show()
    cv2.imwrite(filename="img/"+name+".jpg", img=img)
