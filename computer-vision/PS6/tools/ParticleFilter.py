import cv2
import numpy as np
import time
from numba import jit

DEBUG = False

def score(row, convolved_img, window) -> int:
    (w_v, w_u) = window.shape
    u = row[0, 0]
    v = row[0, 1]
    w_v *= 0.5
    w_u *= 0.5
    # if the particle goes out the boundaries, kill it by assinging high MSE
    if (u < w_u) or (u - w_u > convolved_img.shape[0]) or (v < w_v) or (v - w_v > convolved_img.shape[1]):
        return np.max(convolved_img)
    else:
        return convolved_img[int(u - w_u), int(v - w_v)]

def get_point(row):
    return int(row[0, 1]), int(row[0, 0])


class ParticleFilter:

    def __init__(self, u, v, window, N=200, sigma=10, IIR_alpha=None):
        """
        create a particle filter
        :param u: x location of the initial window
        :param v: y location of the initial window
        :param window: the template to match
        :param N: amount of particles
        :param sigma: smoothness coefficient on the results
        :param IIR_alpha: the alpha parameter use for appearance model update
        """
        self.N = N
        self.particles = np.mat(np.hstack((np.mat([u]*N).T, np.mat([v]*N).T)))
        self.window = window
        self.sigma = sigma
        self.scores = np.array([0] * N)
        self.IIR_alpha = IIR_alpha

    def predict(self):
        self.particles = np.mat(np.random.normal(self.particles, 20))

    def update(self, img):
        self.predict()
        # compute MSE of each particle
        start_time = time.time()
        convolve_img = cv2.matchTemplate(img.copy(), self.window.astype(np.uint8), method=cv2.TM_SQDIFF)
        # debug = np.mat(np.exp(-1. * np.mat(normalize(convolve_img)))) - np.mat(np.exp(2. * (self.sigma ** 2)))
        if DEBUG:
            print("--- %s seconds ---" % (time.time() - start_time))
            cv2.imshow("convolve", normalize(convolve_img))
        scores = np.array(list(map(lambda row: score(row, convolve_img, self.window), self.particles)))
        # convert the results into measurment
        # scores = np.array([max(scores)]*len(scores)) - scores
        scores = np.exp(-scores.astype(float) / (2. * (self.sigma ** 2)))
        # normalize the results
        scores /= np.sum(scores)
        self.scores = scores
        # pick element to survive
        survivors = np.random.choice(a=self.particles.shape[0], size=self.N, replace=True, p=self.scores)
        self.particles = self.particles[survivors, :]
        self.scores = self.scores[survivors]
        self.scores /= np.sum(self.scores)
        if self.IIR_alpha is not None:
            self.update_window(img)

    def update_window(self, img):
        best_index = np.argmax(self.scores)
        row = self.particles[best_index]
        (v, u) = get_point(row) # (int(np.asscalar(np.mat(self.scores) * self.particles[:, 1])), int(np.asscalar(np.mat(self.scores) * self.particles[:, 0])))
        w_u = int(self.window.shape[0] / 2)
        w_v = int(self.window.shape[1] / 2)
        cv2.imshow("window", img[u - w_u:u + w_u, v - w_v:v + w_v])
        cv2.imshow("window_2", self.window.astype(np.uint8))
        self.window = (1. - self.IIR_alpha) * np.mat(img[u - w_u:u + w_u, v - w_v:v + w_v]) \
                      +     self.IIR_alpha  * np.mat(self.window)
        cv2.imshow("window_updated", self.window.astype(np.uint8))
        # cv2.waitKey()
        if DEBUG:
            pass

    def draw_particles(self, img):
        for p in self.particles:
            img = cv2.circle(img, get_point(p), 2, (255, 0, 0))
        return img

    def draw_tracking_window(self, img):
        weighted_u_sum = int(self.scores * self.particles[:, 0])
        weighted_v_sum = int(self.scores * self.particles[:, 1])
        w_u = int(self.window.shape[0] / 2)
        w_v = int(self.window.shape[1] / 2)
        pt1 = (weighted_v_sum - w_v, weighted_u_sum - w_u)
        pt2 = (weighted_v_sum + w_v, weighted_u_sum + w_u)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0))
        cv2.circle(img, (weighted_v_sum, weighted_u_sum), int(np.std(self.particles[:, 0])), (0, 255, 0))
        return img

def normalize(img):
    return (img - np.min(img)) / (np.max(img)-np.min(img))
