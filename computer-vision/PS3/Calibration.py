import numpy as np


def compute_projection_matrix(uv, xyz):
    (h_uv, w_uv) = uv.shape
    (h_xyz, w_xyz) = xyz.shape
    assert h_uv == h_xyz, "lists must have same number of rows"
    everybody = np.hstack((xyz, uv))
    M = np.ones((12))
    for (x, y, z, u, v) in everybody:
        M = np.vstack((
            M,
            [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u],
            [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]
        ))
    M = M[1:, :]
    (u, s, vh) = np.linalg.svd(M)
    return np.reshape(-1 * vh[-1, :], (3, 4))


def compute_residual(uv, xyz, M):
    (h_uv, w_uv) = uv.shape
    (h_xyz, w_xyz) = xyz.shape
    xyz1 = np.hstack((xyz, np.ones((h_xyz, 1))))
    uvs = xyz1 * np.mat(M.T)
    uv_calc = uvs[:, :-1] / uvs[:, -1]
    return np.sum(np.square(uv_calc - uv))


def k_fold(k, uv, xyz):
    everybody = np.hstack((uv, xyz))
    lowest_res = np.inf
    res_sum = 0
    for i in range(100):
        np.random.shuffle(everybody)
        M = compute_projection_matrix(everybody[0:k, 0:2], everybody[0:k, 2:5])
        res = compute_residual(everybody[k:k + 4, 0:2], everybody[k:k + 4, 2:5], M)
        res_sum += res
        if res < lowest_res:
            lowest_res = res
    return lowest_res, res_sum / 100


def find_center(M):
    (h, w) = M.shape
    assert (h, w) == (3, 4)
    Q = np.mat(M[:, 0:3])
    m4 = np.mat(M[:, 3:4])
    return -np.linalg.inv(Q) * m4

def compute_fundamental(uv, uv2):
    everybody = np.hstack((uv, uv2))
    M = np.ones((9))
    for (u, v, u2, v2) in everybody:
        M = np.vstack((
            M,
            [u*u2, u2*v, u2, v2*u, v*v2, v2, u, v, 1]
        ))
    M = M[1:, :]
    (u, s, vh) = np.linalg.svd(M)
    F = np.reshape(vh[-1, :], (3, 3))
    (u, d, vh) = np.linalg.svd(F)  # note : (np.mat(u)*np.mat(np.diag(d))*np.mat(vh) = F
    return np.mat(u)*np.mat(np.diag([d[0], d[1], d[2]]))*np.mat(vh)


if __name__ == "__main__":
    uv = np.loadtxt("pts2d-norm-pic_a.txt")
    uv1 = np.loadtxt("pts2d-pic_a.txt")
    uv2 = np.loadtxt("pts2d-pic_b.txt")
    xyz = np.loadtxt("pts3d-norm.txt")
    M = compute_projection_matrix(uv, xyz)
    print compute_residual(uv, xyz, M)
    for k in [8, 12, 16]:
        print k_fold(k, uv, xyz)
    print find_center(M)
    F = compute_fundamental(uv1, uv2)
    print F
    print np.mat(np.hstack((uv2[0, :], 1))) * np.mat(F) * np.mat(np.hstack((uv1[0, :], 1))).T
