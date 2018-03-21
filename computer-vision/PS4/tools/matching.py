import numpy as np
import cv2
from matplotlib import pyplot as plt


def match(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return flann.knnMatch(des1, des2, k=2)


def filter_matches(matches):
    matches = sorted(matches, key=lambda x: x[0].distance)
    out = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            out.append((m, n))
    return out


def display_matches(img1, img2, kp1, kp2, matches, amount=30):
    (height, width) = img1.shape
    out = np.hstack((img1, img2))
    matches = filter_matches(matches)
    for i, (m, n) in enumerate(matches[:amount]):
            (pt1x, pt1y) = kp1[m.queryIdx].pt
            (pt2x, pt2y) = kp2[m.trainIdx].pt
            out = cv2.line(out, (int(pt1x), int(pt1y)), (int(pt2x + width), int(pt2y)), 0, thickness=2)
    return out


if __name__ == '__main__':
    from tools import sift
    import pylab

    imgA = cv2.imread('inputs/simA.jpg')[:, :, 2]
    sift_kp_A, sift_desc_A = sift.get_sift_features(imgA)
    imgB = cv2.imread('inputs/simB.jpg')[:, :, 2]
    sift_kp_B, sift_desc_B = sift.get_sift_features(imgB)
    matches = match(sift_desc_A, sift_desc_B)
    putative_img = display_matches(imgA, imgB, sift_kp_A, sift_kp_B, matches)
    pylab.imshow(putative_img, cmap=pylab.gray())
    pylab.show()
