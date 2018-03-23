import random

import cv2
import numpy as np
from numba import jit
from tools import matching, sift
from tools.matching import match


def prepare_matches_for_ransac(kp1, kp2, matches, clean=True):
    if clean:
        matches = matching.filter_matches(matches)
    return list(map(lambda elem: (kp1[elem[0].queryIdx].pt, kp2[elem[0].trainIdx].pt), matches))


def compute_model_trans(pairseq):
    dx = 0
    dy = 0
    i = 0
    for pair in pairseq:
        ((u1, v1), (u12, v12)) = pair
        dx += u12 - u1
        dy += v12 - v1
        i += 1
    dx = dx / i
    dy = dy / i
    return np.mat(
        [
            [1, 0, dx],
            [0, 1, dy],
        ]
    )


def compute_model_sim(pairseq):
    Mpairs = []
    uvpair = []
    for pair in pairseq:
        ((u1, v1), (u12, v12)) = pair
        Mpairs.append(np.mat(
            [
                [u1, -v1, 1, 0],
                [v1, u1, 0, 1]
            ]
        ))
        uvpair.append(np.mat([[u12], [v12]]))
    M = np.vstack(Mpairs)
    uv = np.vstack(uvpair)
    A = np.linalg.lstsq(M, uv)[0]
    return np.mat(
        [
            [A[0, 0], -A[1, 0], A[2, 0]],
            [A[1, 0], A[0, 0], A[3, 0]],
        ]
    )


def vote(elem, model, eps):
    ((u1, v1), (u12, v12)) = elem
    return np.max(np.abs((model * np.mat([[u1], [v1], [1]])) - np.mat([[u12], [v12]]))) < eps


def ransac(vote_func, compute_model_func, prepared_matches, s, p=0.9, eps=10):
    N = np.inf
    sample_count = 0
    e = 1.0
    best = 0
    best_param = None
    number_of_point = len(prepared_matches)
    while N > sample_count:
        # choose sample and count inliers
        # choose sample
        np.random.shuffle(prepared_matches)
        samples = prepared_matches[0:s]
        model = compute_model_func(samples)
        # count inlier
        score = sum(list(map(lambda elem: vote_func(elem, model, eps=eps), prepared_matches)))
        # save best result
        if score > best:
            best = score
            best_param = model
        #
        e0 = 1 - (score / number_of_point)
        # adapt number of sample
        if e0 < e:
            e = e0
            N = np.log(1 - p) / np.log(1 - (1 - e) ** s)
        sample_count = sample_count + 1
    best_inliers = []
    for el in prepared_matches:
        if vote_func(el, best_param, eps=eps):
            best_inliers.append(el)
    return best, compute_model_func(best_inliers)


def superpose_images(imgA, M, imgB):
    out = cv2.warpAffine(imgA, M, (imgB.shape[1], imgB.shape[0]))
    for x, y in np.ndindex(out.shape):
        out[x, y] = int(0.5 * out[x, y] + 0.5 * imgB[x, y])
    return out


if __name__ == '__main__':
    imgA = cv2.imread('inputs/transA.jpg')[:, :, 2]
    sift_kp_A, sift_desc_A = sift.get_sift_features(imgA)
    imgB = cv2.imread('inputs/transB.jpg')[:, :, 2]
    sift_kp_B, sift_desc_B = sift.get_sift_features(imgB)
    matches = match(sift_desc_A, sift_desc_B)
    prepared_matches = prepare_matches_for_ransac(sift_kp_A, sift_kp_B, matches, clean=True)
    M = ransac(vote, compute_model_trans, prepared_matches, s=1, p=0.9, eps=10)
    print(M)
    imgA2 = cv2.imread('inputs/simA.jpg')[:, :, 2]
    sift_kp_A2, sift_desc_A2 = sift.get_sift_features(imgA2)
    imgB2 = cv2.imread('inputs/simB.jpg')[:, :, 2]
    sift_kp_B2, sift_desc_B2 = sift.get_sift_features(imgB2)
    matches2 = match(sift_desc_A2, sift_desc_B2)
    prepared_matches2 = prepare_matches_for_ransac(sift_kp_A2, sift_kp_B2, matches2, clean=True)
    M2 = ransac(vote, compute_model_sim, prepared_matches2, s=2, p=0.9, eps=10)
    print(M2)
