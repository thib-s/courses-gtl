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


@jit(nopython=True)
def vote(elem, u, v, eps=10):
    ((x1, y1), (x2, y2)) = elem
    u1 = x1 - x2
    v1 = y1 - y2
    if (abs(u1 - u) < eps) and (abs(v1 - v) < eps):
        return 1
    else:
        return 0


def ransac_trans(prepared_matches):
    N = np.inf
    sample_count = 0
    e = 1.0
    s = 1
    p = 0.9
    best = 0
    best_param = None
    number_of_point = len(prepared_matches)
    while N > sample_count:
        # choose sample and count inliers
        # choose sample
        [(x1, y1), (x2, y2)] = random.choice(prepared_matches)
        u = x1 - x2
        v = y1 - y2
        # count inlier
        score = sum(list(map(lambda elem: vote(elem, u, v), prepared_matches)))
        # save best result
        if score > best:
            best = score
            best_param = (u, v)
        #
        e0 = 1 - (score / number_of_point)
        # adapt number of sample
        if e0 < e:
            e = e0
            N = np.log(1 - p) / np.log(1 - (1 - e) ** s)
        sample_count = sample_count + 1
    return best, best_param


if __name__ == '__main__':
    imgA = cv2.imread('inputs/simA.jpg')[:, :, 2]
    sift_kp_A, sift_desc_A = sift.get_sift_features(imgA)
    imgB = cv2.imread('inputs/simB.jpg')[:, :, 2]
    sift_kp_B, sift_desc_B = sift.get_sift_features(imgB)
    matches = match(sift_desc_A, sift_desc_B)
    prepared_matches = prepare_matches_for_ransac(sift_kp_A, sift_kp_B, matches, clean=True)
    print(ransac_trans(prepared_matches))
