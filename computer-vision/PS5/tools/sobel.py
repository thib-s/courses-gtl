import cv2
import numpy as np


def compute_gradients(img1, blursize=1, kersize=3):
    # kernel_smooth = np.vectorize(lambda x: x * 0.1)(np.array(
    #     [
    #         [1, 2, 1],
    #         [2, 4, 2],
    #         [1, 2, 1]
    #     ]
    # )
    # )
    img1 = cv2.GaussianBlur(src=img1,
                            ksize=(4*blursize+1, 4*blursize+1),
                            sigmaX=blursize, sigmaY=blursize,
                            borderType=cv2.BORDER_REFLECT)
    # img1 = cv2.filter2D(img1, -1, kernel_smooth, borderType=cv2.BORDER_CONSTANT)

    # kernel = np.vectorize(lambda x: x * 0.125)(np.array(
    #     [
    #         [-1, 0, 1],
    #         [-2, 0, 2],
    #         [-1, 0, 1]
    #     ]
    # ))
    # kernel_h = np.vectorize(lambda x: x * 0.125)(np.array(
    #     [
    #         [-1, -2, -1],
    #         [0, 0, 0],
    #         [1, 2, 1]
    #     ]
    # ))

    img_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=kersize)
    # cv2.filter2D(img1, -1, kernel, borderType=cv2.BORDER_REFLECT)
    img_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=kersize)
    # cv2.filter2D(img1, -1, kernel_h, borderType=cv2.BORDER_REFLECT)
    return img_x, img_y


def get_sobel(img_x, img_y):
    vfunc = np.vectorize(lambda t: t ** 2)

    img = np.vectorize(np.math.sqrt)(np.add(
        vfunc(img_x),
        vfunc(img_y)
    )
    )
    return img


def normalize(img):
    mini = np.min(img)
    maxi = np.max(img)
    return np.vectorize(lambda x: int(255 * (x - mini)/(maxi - mini)))(img)
