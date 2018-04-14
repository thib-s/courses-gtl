import cv2
import numpy as np


class Particle:

    def __init__(self, state: np.mat):
        self.state: np.mat = state

    def score(self, img, convolved_img) -> int:
        u = self.state[0, 0]
        v = self.state[1, 0]
        # if the particle goes out the boundaries, kill it by assinging high MSE
        if (u < 0) or (u > img.shape(0)) or (v < 0) or (v > img.shape(1)):
            return np.max(convolved_img)
        else:
            return convolved_img[u, v]

    def add_noise(self, sigma):
        self.state = np.mat(np.random.normal(self.state, sigma))


class ParticleFilter:

    def __init__(self, init_state: np.mat, window, N=200):
        self.N = N
        self.particles = [Particle(init_state)] * N
        self.window = window

    def predict(self):
        map(lambda p: p.add_noise, self.particles)

    def update(self, img):
        # compute MSE of each particle
        convolve_img = cv2.filter2D(img, -1, self.window, borderType=cv2.BORDER_REFLECT)
        scores = np.mat(map(lambda p: p.score(img, convolve_img), self.particles))
        # convert the results into measurment
        sigma = np.var(scores)
        scores = np.exp(-1 * scores / (2 * (sigma ** 2)))
        # normalize the results
        scores /= np.sum(scores)
        # pick element to survive
        indices = np.random.choice(len(self.particles), size=self.N, replace=True, p=scores)
        survivor = self.particles[indices]
        # add noise to survivors
        self.particles = survivor

    def draw_particles(self, img):
        for p in self.particles:
            img = cv2.circle(img, p.state[0, 0], 2, (255, 0, 0))
        return img

    def draw_tracking_window(self, img):
        scores = np.mat(map(lambda p: p.score(img, self.window), self.particles))
        # convert the results into measurment
        sigma = np.var(scores)
        scores = np.exp(-1 * scores / (2 * (sigma ** 2)))
        # normalize the results
        scores /= np.sum(scores)
        weighted_u_sum = 0
        weighted_v_sum = 0
        for i in range(len(scores)):
            weighted_u_sum += scores * self.particles[i].state[0]
            weighted_v_sum += scores * self.particles[i].state[1]
        pt1 = (weighted_u_sum - 10, weighted_v_sum - 10)
        pt2 = (weighted_u_sum + 10, weighted_v_sum + 10)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0))
        cv2.circle(img, (weighted_u_sum, weighted_v_sum), sigma, (0, 255, 0))
