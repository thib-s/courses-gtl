from math import ceil

import cv2


def reduce(img):
    """
    reduce stage of the pyramid
    :param img: the image to reduce
    :return: the reduced image
    """
    img = cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)


def expand(img, shape=None):
    """
    expand stage of the pyramid
    :param img: the image to expand
    :return: the expanded image
    """
    if shape is None:
        shape = (img.shape[0]*2, img.shape[1]*2)
    img = cv2.resize(img, (shape[1], shape[0]))
    return cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_REFLECT)


def compute_pyramid(img, depth):
    """
    compute the Laplacian pyramid
    :param img: the originla image
    :param depth: amount of lagrangian to compute
    :return: the list of the ordered Lagrangians plus the list of ordered Gaussians
    """
    G_i = [img]
    L_i = []
    for i in range(depth):
        G_i.append(reduce(G_i[-1]))
        L_i.append(expand(G_i[-1], G_i[-2].shape) - G_i[-2])
    return L_i, G_i


if __name__ == '__main__':
    img = cv2.imread('images/DataSeq1/yos_img_01.jpg')
    cv2.imshow("reduce", reduce(img))
    cv2.waitKey()
    cv2.imshow("expand", expand(img))
    cv2.waitKey()
    cv2.imshow("laplacian", img[0:64, 0:80] - expand(reduce(img)))
    cv2.waitKey()
