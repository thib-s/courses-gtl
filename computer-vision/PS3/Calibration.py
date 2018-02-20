import numpy as np


def compute_projection_matrix(uv, xyz):
    (h_uv, w_uv) = uv.shape
    (h_xyz, w_xyz ) = xyz.shape
    assert h_uv == h_xyz, "lists must have same number of rows"
    everybody = np.hstack((xyz, uv))
    M = np.ones((12))
    for (x, y, z, u, v) in everybody:
        M = np.vstack((
            M,
            [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u],
            [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
        ))
    M = M[1:, :]
    (u, s, vh) = np.linalg.svd(M)
    return np.reshape(-1*vh[-1, :], (3, 4))


def compute_residual(uv, xyz, M):
    (h_uv, w_uv) = uv.shape
    (h_xyz, w_xyz ) = xyz.shape
    xyz1 = np.hstack((xyz, np.ones((h_xyz, 1))))
    uvs = xyz1 * np.mat(M.T)
    uv_calc = uvs[:, :-1] / uvs[:, -1]
    return np.sum(np.square(uv_calc - uv))


if __name__ == "__main__":
    uv = np.loadtxt("pts2d-norm-pic_a.txt")
    xyz = np.loadtxt("pts3d-norm.txt")
    M = compute_projection_matrix(uv, xyz)
    print compute_residual(uv, xyz, M)
