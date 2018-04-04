import cv2
import numpy as np
from numba import jit, cuda
from tools import sobel, laplacian_pyramid


def compute_lk(grad_x, grad_y, grad_t, wd):
    """
    compute the LK optical flow without using CUDA
    :param grad_x: x gradient of image
    :param grad_y: y gradient of image
    :param grad_t: temporal gradient of image
    :param wd: the window size
    :return: U and V corresponding to optical flow
    """
    U = np.zeros(grad_x.shape, dtype=float)
    V = np.zeros(grad_x.shape, dtype=float)
    R = np.zeros_like(grad_x)  # todo: set the correct size for r
    (width, height) = grad_x.shape
    assert (width, height) == grad_y.shape, "gradient images must be the same size!"
    for i in range(wd, width - wd, 1):
        for j in range(wd, height - wd, 1):
            Ixx = float(0.0)
            Ixy = float(0.0)
            Iyy = float(0.0)
            Ixt = float(0.0)
            Iyt = float(0.0)
            for x in range(-wd, +wd, 1):
                for y in range(-wd, +wd, 1):
                    Ixt += grad_x[i + x, j] * grad_t[i + x, j + y]
                    Iyt += grad_y[i + x, j] * grad_t[i + x, j + y]
                    Ixx += grad_x[i + x, j + y] ** 2
                    Ixy += grad_x[i + x, j + y] * grad_y[i + x, j + y]
                    Iyy += grad_y[i + x, j + y] ** 2
            norm = float(Ixx * Iyy) - float(Ixy ** 2)
            if Ixx == 0.0 and Ixy == 0.0 and Iyy == 0.0:
                U[i, j] = float(0.0)
                V[i, j] = float(0.0)
            else:
                U[i, j] = float(float(-Ixt * Iyy + Iyt * Ixy) / float(norm))
                V[i, j] = float(float(Ixt * Ixy - Iyt * Ixx) / float(norm))
    return U, V


@cuda.jit
def compute_lk_cuda(grad_x, grad_y, grad_t, U, V, wd):
    """
    CUDA kernel computing the optical flow
    :param grad_x: x gradient of image
    :param grad_y: y gradient of image
    :param grad_t: temporal gradient of image
    :param wd: the window size
    """
    i = cuda.blockIdx.x + wd
    j = cuda.blockIdx.y + wd
    Ixx = 0.0
    Ixy = 0.0
    Iyy = 0.0
    Ixt = 0.0
    Iyt = 0.0
    (width, height) = grad_x.shape
    for x in range(max(i-wd, 0), min(i+wd, width), 1):
        for y in range(max(j-wd, 0), min(j+wd, height), 1):
            Ixt += grad_x[x, j] * grad_t[x, y]
            Iyt += grad_y[x, j] * grad_t[x, y]
            Ixx += grad_x[x, y] ** 2
            Ixy += grad_x[x, y] * grad_y[x, y]
            Iyy += grad_y[x, y] ** 2
    norm = float(Ixx * Iyy) - float(Ixy ** 2)
    if Ixx == 0.0 and Ixy == 0.0 and Iyy == 0.0:
        U[i, j] = float(0.0)
        V[i, j] = float(0.0)
    else:
        U[i, j] = float(-Ixt * Iyy + Iyt * Ixy) / norm
        V[i, j] = float(Ixt * Ixy - Iyt * Ixx) / norm


@jit
def wrapper_compute_lk_cuda(grad_x, grad_y, grad_t, wd):
    """
    wrapper to compute LK flow using CUDA
    :param grad_x: x gradient of image
    :param grad_y: y gradient of image
    :param grad_t: temporal gradient of image
    :param wd: the window size
    :return: U and V corresponding to optical flow
    """
    U = np.zeros(grad_x.shape, dtype=float)
    V = np.zeros(grad_x.shape, dtype=float)
    blockspergrid_x = grad_x.shape[0] - 2 * wd
    blockspergrid_y = grad_y.shape[1] - 2 * wd
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    compute_lk_cuda[blockspergrid, 1](
        grad_x,
        grad_y,
        grad_t,
        U,
        V,
        wd
    )
    return U, V


def warp_img(img, U, V):
    """
    warp the image according to the U and V optical flow vectors
    :param img: the image to warp
    :param U:
    :param V:
    :return: the warped image
    """
    (width, height) = U.shape
    mapU = np.zeros((width, height), dtype=np.float32)
    mapV = np.zeros((width, height), dtype=np.float32)
    for x in range(0, width):
        for y in range(0, height):
            mapU[x, y] = U[x, y] + y
            mapV[x, y] = V[x, y] + x
    return cv2.remap(src=np.transpose(img), map1=mapV, map2=mapU, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def LK_pyramid(img: dict, level, wd):
    """
    compute the hierachical LK optical flow, this function must be called trough wrapper_LK_pyramid
    :param img: dictionnary containing various gradient used
    :param level: the remaining number of steps
    :param wd: the windows radius used in LK
    :return: the U and V vector corresponding to the optical flow
    """
    # if we are on a leaf node
    if level == 0:
        # LK computation
        return wrapper_compute_lk_cuda(img['grad_x'], img['grad_y'], img['grad_t'], int(wd))
    # if we are not on a leaf node
    img_red = {}
    # step 1: reduce all gradients images
    for key in img.keys():
        img_red[key] = laplacian_pyramid.reduce(img[key])
    # step 2: recursive call, get the result from the previous stages and expand it
    (u, v) = LK_pyramid(img_red, level - 1, wd / 2)
    u = laplacian_pyramid.expand(u)
    v = laplacian_pyramid.expand(v)
    # step 3 : warping
    img_warped = {}
    for key in img.keys():
        img_warped[key] = warp_img(img[key], u, v)
    # step 4 : current stage LK
    (current_u, current_v) = wrapper_compute_lk_cuda(
        img_warped['grad_x'],
        img_warped['grad_y'],
        img_warped['grad_t'],
        int(wd))
    # step 5 : add current result to the accumulator and return it
    return current_u + u, current_v + v


def wrapper_LK_pyramid(img0, img1, depth, wd):
    """
    compute the hierarchical Lk optic flow of an image using CUDA
    :param img0: the image at t
    :param img1: the image at t+1
    :param depth: the depth of the pyramid
    :param wd: the size of the window
    :return: the U and V corresponding to optical flow
    """
    grad_t = img1 - img0
    grad_t = grad_t.astype(float)
    grad_x, grad_y = sobel.compute_gradients(img0)
    img = {
        'grad_x': grad_x,
        'grad_y': grad_y,
        'grad_t': grad_t,
    }
    return LK_pyramid(img, depth, wd)


if __name__ == '__main__':
    im0 = cv2.imread("images/TestSeq/Shift0.png")[:, :, 0]
    im1 = cv2.imread("images/TestSeq/ShiftR5U5.png")[:, :, 0]
    im0 = cv2.GaussianBlur(src=im0, ksize=(7, 7), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REFLECT)
    im1 = cv2.GaussianBlur(src=im1, ksize=(7, 7), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_REFLECT)
    grad_t = im1 - im0
    grad_t = grad_t.astype(float)
    grad_x, grad_y = sobel.compute_gradients(im0)
    U, V = wrapper_compute_lk_cuda(grad_x, grad_y, grad_t, 20)
    warped = warp_img(im0, U, V)
    cv2.imshow("warped image", np.hstack((im0, warped, im1)))
    cv2.waitKey()
    U2, V2 = wrapper_LK_pyramid(im0, im1, 2, 20)
    warped = warp_img(im0, U2, V2)
    cv2.imshow("warped image pyramid", np.hstack((im0, warped, im1)))
    cv2.waitKey()
