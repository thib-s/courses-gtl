import numpy as np


def w(x, y):
    return 1


def compute_harris(grad_x, grad_y, windows_size, window_funct=w, alpha=0.05):
    wd = int(windows_size / 2)
    R = np.zeros_like(grad_x)  # todo: set the correct size for r
    (width, height) = grad_x.shape
    assert (width, height) == grad_y.shape, "gradient images must be the same size!"
    for i in range(wd, width - wd, 1):
        for j in range(wd, width - wd, 1):
            M = np.mat([[0, 0], [0, 0]])
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
