import cv2
import pylab
import numpy as np
from numba import jit, cuda
from tools import sobel


def compute_lk(grad_x, grad_y, grad_t, wd):
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


@cuda.jit  # (float[:, :], float[:, :], float[:, :], float[:, :], float[:, :], uint8)
def compute_lk_cuda(grad_x, grad_y, grad_t, U, V, wd):
    i = cuda.blockIdx.x + wd
    j = cuda.blockIdx.y + wd
    Ixx = 0.0
    Ixy = 0.0
    Iyy = 0.0
    Ixt = 0.0
    Iyt = 0.0
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
        U[i, j] = float(-Ixt * Iyy + Iyt * Ixy) / norm
        V[i, j] = float(Ixt * Ixy - Iyt * Ixx) / norm


@jit
def wrapper_compute_lk_cuda(grad_x, grad_y, grad_t, wd):
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
    (width, height) = U.shape
    mapU = np.zeros((height, width), dtype=np.float32)
    mapV = np.zeros((height, width), dtype=np.float32)
    for x in range(0, width):
        for y in range(0, height):
            mapU[y, x] = U[x, y] + x
            mapV[y, x] = V[x, y] + y
    return cv2.remap(src=img, map1=mapU, map2=mapV, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


if __name__ == '__main__':
    im0 = cv2.imread("images/TestSeq/Shift0.png")[:, :, 0]
    im1 = cv2.imread("images/TestSeq/ShiftR2.png")[:, :, 0]
    im0 = cv2.GaussianBlur(src=im0, ksize=(19, 7), sigmaX=10, sigmaY=3, borderType=cv2.BORDER_REFLECT)
    im1 = cv2.GaussianBlur(src=im1, ksize=(19, 7), sigmaX=10, sigmaY=3, borderType=cv2.BORDER_REFLECT)
    grad_t = im1 - im0
    grad_t = grad_t.astype(float)
    grad_x, grad_y = sobel.compute_gradients(im0)
    U, V = wrapper_compute_lk_cuda(grad_x, grad_y, grad_t, 5)
    pylab.imshow(U, cmap=pylab.gray())
    pylab.imshow(V, cmap=pylab.gray())
    pylab.show()
    cv2.imshow("warped image", warp_img(im0, U, V))
    cv2.waitKey()
