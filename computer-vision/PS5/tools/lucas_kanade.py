import cv2
from numba import cuda
from numba.cuda import jit, np
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


@jit
def wrapper_compute_lk_cuda(grad_x, grad_y, grad_t, wd):
    U = np.zeros(grad_x.shape, dtype=float)
    V = np.zeros(grad_x.shape, dtype=float)
    blockspergrid_x = grad_x.shape[0] - 2 * wd
    blockspergrid_y = grad_y.shape[1] - 2 * wd
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    compute_lk_cuda[blockspergrid, 1](
        grad_x.astype(float),
        grad_y.astype(float),
        grad_t.astype(float),
        U.astype(float),
        V.astype(float),
        int(wd)
    )
    return U, V


def warp_img(img, U, V):
    pass


if __name__ == '__main__':
    im0 = cv2.imread("images/TestSeq/Shift0.png")[:, :, 0]
    im1 = cv2.imread("images/TestSeq/ShiftR2.png")[:, :, 0]
    grad_t = im1 - im0
    grad_t = grad_t.astype(float)
    grad_x, grad_y = sobel.compute_gradients(im0)
    U, V = compute_lk(grad_x, grad_y, grad_t, 5)
    cv2.imshow("U", U)
    cv2.imshow("V", V)
    cv2.waitKey()
