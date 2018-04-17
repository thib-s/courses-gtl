import cv2
import numpy as np
import time
from numba import jit

DEBUG = False


def score(row, convolved_imgs, window, scale_bins) -> int:
    w_v = window.shape[0]
    w_u = window.shape[1]
    u = row[0, 0]
    v = row[0, 1]
    s = row[0, 2]
    convolved_img = convolved_imgs[s]
    w_v *= scale_bins[s] * 0.5
    w_u *= scale_bins[s] * 0.5
    # if the particle goes out the boundaries, kill it by assinging high MSE
    if (u <= w_u) or (u - w_u >= convolved_img.shape[0]) or (v <= w_v) or (v - w_v >= convolved_img.shape[1]):
        return np.max(convolved_img) / scale_bins[s]**2
    else:
        return convolved_img[int(u - w_u), int(v - w_v)] / scale_bins[s]**2


def get_point(row):
    return int(row[0, 1]), int(row[0, 0])


class ParticleFilter:

    def __init__(self, u, v, window, N=200, sigma=10, IIR_alpha=None, scale_bins=None):
        """
        create a particle filter
        :param u: x location of the initial window
        :param v: y location of the initial window
        :param window: the template to match
        :param N: amount of particles
        :param sigma: smoothness coefficient on the results
        :param IIR_alpha: the alpha parameter use for appearance model update
        """
        if scale_bins is None:
            scale_bins = [1.]
        self.N = N
        self.particles = np.mat(np.hstack((np.mat([u] * N).T, np.mat([v] * N).T, np.mat([0] * N).T)))
        self.window = window
        self.sigma = sigma
        self.scores = np.array([0] * N)
        self.IIR_alpha = IIR_alpha
        self.scale_bins = scale_bins
        self.freeze = False

    def predict(self):
        if self.freeze:
            noise = 1
        else:
            noise = 20
        self.particles[:, 0:2] = np.mat(np.random.normal(self.particles[:, 0:2], noise))
        if len(self.scale_bins) > 1:
            # 0.1 has been chosen because we assume that it take between 10 and 20 frame for windows to decrease 10%
            to_increment = np.random.choice(range(self.particles.shape[0]), int(self.particles.shape[0] * 0.1))
            to_decrement = np.random.choice(range(self.particles.shape[0]), int(self.particles.shape[0] * 0.1))
            self.particles[to_increment, 2] = list(map(lambda id: max(0, min(len(self.scale_bins)-1, self.particles[id, 2] + 1)), to_increment))
            self.particles[to_decrement, 2] = list(map(lambda id: max(0, min(len(self.scale_bins)-1, self.particles[id, 2] - 1)), to_decrement))

    def update(self, img):
        self.predict()
        # compute MSE of each particle
        start_time = time.time()
        convolve_imgs = list(map(
            lambda scale: cv2.matchTemplate(
                img.copy(),
                cv2.resize(self.window.astype(np.uint8), (0, 0), fx=scale, fy=scale),
                method=cv2.TM_SQDIFF),
            self.scale_bins))
        # debug = np.mat(np.exp(-1. * np.mat(normalize(convolve_img)))) - np.mat(np.exp(2. * (self.sigma ** 2)))
        if DEBUG:
            print("--- %s seconds ---" % (time.time() - start_time))
            #  cv2.imshow("convolve", normalize(convolve_img))
        scores = np.array(
            list(map(lambda row: score(row, convolve_imgs, self.window, self.scale_bins), self.particles)))
        # convert the results into measurment
        if np.min(scores)/(self.window.shape[0]*self.window.shape[1]*self.window.shape[2]) > 3E3:
            self.freeze = True
        else:
            self.freeze = False
        # print(np.min(scores)/(self.window.shape[0]*self.window.shape[1]*self.window.shape[2]))
        if not self.freeze:
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
        (v, u) = get_point(
            row)  # (int(np.asscalar(np.mat(self.scores) * self.particles[:, 1])), int(np.asscalar(np.mat(self.scores) * self.particles[:, 0])))
        s = self.scale_bins[row[0, 2]]
        w_u = int(s * self.window.shape[0] / 2)
        w_v = int(s * self.window.shape[1] / 2)
        if DEBUG:
            cv2.imshow("window", img[u - w_u:u + w_u, v - w_v:v + w_v, :])
            cv2.imshow("window_2", self.window.astype(np.uint8))
        for i in range(3):
            self.window[:,:,i] = (1. - self.IIR_alpha) * np.mat(cv2.resize(img[u - w_u:u + w_u, v - w_v:v + w_v, i], (self.window.shape[0], self.window.shape[1]))) \
                          + self.IIR_alpha * np.mat(self.window[:,:,i])
        if DEBUG:
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
        weighted_s_sum = int(self.scores * self.particles[:, 2])
        w_u = int(self.scale_bins[int(weighted_s_sum)] * self.window.shape[0] / 2)
        w_v = int(self.scale_bins[int(weighted_s_sum)] * self.window.shape[1] / 2)
        pt1 = (weighted_v_sum - w_v, weighted_u_sum - w_u)
        pt2 = (weighted_v_sum + w_v, weighted_u_sum + w_u)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0))
        cv2.circle(img, (weighted_v_sum, weighted_u_sum), int(np.std(self.particles[:, 0])), (0, 255, 0))
        return img


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))
