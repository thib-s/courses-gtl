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
    best = 0
    best_param = None
    for i in range(int(len(prepared_matches) / 2)):
        [(x1, y1), (x2, y2)] = random.choice(prepared_matches)
        u = x1 - x2
        v = y1 - y2
        score = sum(list(map(lambda elem: vote(elem, u, v), prepared_matches)))
        if score > best:
            best = score
            best_param = (u, v)
    return best, best_param


if __name__ == '__main__':
    imgA = cv2.imread('inputs/simA.jpg')[:, :, 2]
    sift_kp_A, sift_desc_A = sift.get_sift_features(imgA)
    imgB = cv2.imread('inputs/simB.jpg')[:, :, 2]
    sift_kp_B, sift_desc_B = sift.get_sift_features(imgB)
    matches = match(sift_desc_A, sift_desc_B)
    prepared_matches = prepare_matches_for_ransac(sift_kp_A, sift_kp_B, matches, clean=True)
    print(ransac_trans(prepared_matches))
