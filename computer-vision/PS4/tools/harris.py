import numpy as np
import time
import cv2
from numba import jit, cuda


@jit
def w(x, y):
    return 1


@jit
def compute_harris(grad_x, grad_y, windows_size, window_funct=w, alpha=0.05):
    wd = int(windows_size / 2)
    R = np.zeros_like(grad_x)  # todo: set the correct size for r
    (width, height) = grad_x.shape
    assert (width, height) == grad_y.shape, "gradient images must be the same size!"
    for i in range(wd, width - wd, 1):
        for j in range(wd, height - wd, 1):
            M = np.array([[0, 0], [0, 0]])
            for x in range(-wd, +wd, 1):
                for y in range(-wd, +wd, 1):
                    M = M + window_funct(x, y) * np.mat(
                        [
                            [grad_x[i + x, j + y] ** 2, grad_x[i + x, j + y] * grad_y[i + x, j + y]],
                            [grad_x[i + x, j + y] * grad_y[i + x, j + y], grad_y[i + x, j + y] ** 2]
                        ]
                    )
            R[i, j] = np.linalg.det(M) - alpha * np.trace(M) ** 2
    return R


@cuda.jit
def compute_harris_cuda(grad_x, grad_y, R, wd, alpha):
    i = cuda.blockIdx.x + wd
    j = cuda.blockIdx.y + wd
    M1 = 0.0
    M2 = 0.0
    M3 = 0.0
    for x in range(-wd, +wd, 1):
        for y in range(-wd, +wd, 1):
            M1 += grad_x[i + x, j + y] ** 2
            M2 += grad_x[i + x, j + y] * grad_y[i + x, j + y]
            M3 += grad_y[i + x, j + y] ** 2
    R[i, j] = float(M1 * M3 - M2 ** 2) - alpha * float(M1 + M3) ** 2


@jit
def harris_cuda_caller(grad_x, grad_y, wd, alpha):
    R = np.zeros(grad_x.shape)
    blockspergrid_x = grad_x.shape[0] - 2 * wd
    blockspergrid_y = grad_y.shape[1] - 2 * wd
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    compute_harris_cuda[blockspergrid, 1](grad_x, grad_y, R, wd, alpha)
    return R


def get_maximas(harris_img, ws):
    harris_img = harris_img.copy()
    maxi = np.max(harris_img)
    current_max = maxi
    poi = []
    while current_max > (0.1 * maxi):
        current_max = harris_img.max()
        (x, y) = np.argwhere(harris_img.max() == harris_img)[0]
        harris_img[x - ws:x + ws, y - ws:y + ws] = 0
        poi.append((x, y))
    return poi


def print_maximas(src, poi):
    img = src.copy()
    for (x, y) in poi:
        cv2.circle(img, (y, x), 5, (0, 0, 255), -1)
    return img


if __name__ == '__main__':
    import tools.sobel as sobel
    import pylab

    img = cv2.imread('inputs/simA.jpg')[:, :, 2]
    I_x, I_y = sobel.compute_gradients(img, blursize=1, kersize=5 )
    begin = time.time()
    harris_img = harris_cuda_caller(I_x, I_y, wd=10, alpha=0.05)
    harris_img = np.abs(harris_img)
    print("computation time: ", time.time() - begin)
    pylab.imshow(harris_img, cmap=pylab.gray())
    pylab.show()
    poi = get_maximas(harris_img, 10)
    print("number of points:", len(poi))
    pylab.imshow(print_maximas(img, poi))
    pylab.show()
