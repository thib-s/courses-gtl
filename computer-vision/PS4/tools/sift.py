import cv2
import pylab
import tools.harris as harris


def get_sift_features(src, poi=None):
    sift = cv2.xfeatures2d.SIFT_create()
    # des = sift.compute(src, poi, None) # WHY THIS DON'T WANNA WORK ?????
    kp, des = sift.detectAndCompute(src, None)
    return kp, des


if __name__ == '__main__':
    img = cv2.imread('inputs/simA.jpg')[:, :, 2]
    sift_kp, sift_desc = get_sift_features(img)
    pylab.imshow(harris.draw_keypoints(img, sift_kp))
    pylab.show()
