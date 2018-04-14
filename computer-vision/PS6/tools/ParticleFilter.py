import cv2
import numpy as np


class Particle:

    def __init__(self, state: np.mat):
        self.state: np.mat = np.matrix(state)

    def score(self, convolved_img, window) -> int:
        (w_v, w_u) = window.shape
        w_v *= 0.5
        w_u *= 0.5
        u = self.state[0, 0]
        v = self.state[1, 0]
        # if the particle goes out the boundaries, kill it by assinging high MSE
        if (u < w_u) or (u > convolved_img.shape[0]) or (v < w_v) or (v > convolved_img.shape[1]):
            return np.max(convolved_img)
        else:
            return convolved_img[int(u-w_u), int(v-w_v)]

    def add_noise(self, sigma):
        self.state = np.matrix(np.random.normal(self.state, sigma))
        return self

    def get_point(self):
        return int(self.state[0, 0]), int(self.state[1, 0])


class ParticleFilter:

    def __init__(self, init_state: np.mat, window, N=200, sigma=10):
        self.N = N
        self.particles = []
        for i in range(N):
            self.particles.append(Particle(init_state))
        self.window = window
        self.sigma = sigma
        self.scores = [0] * N

    def predict(self):
        self.particles = list(map(lambda p: p.add_noise(5), self.particles))

    def update(self, img):
        self.predict()
        # compute MSE of each particle
        convolve_img = cv2.matchTemplate(img, self.window, method=cv2.TM_SQDIFF_NORMED)
        # debug = np.mat(np.exp(-1. * np.mat(normalize(convolve_img)))) - np.mat(np.exp(2. * (self.sigma ** 2)))
        cv2.imshow("convolve", normalize(convolve_img))
        scores = np.mat(list(map(lambda p: p.score(convolve_img, self.window), self.particles)))
        # convert the results into measurment
        sigma = np.var(scores)
        scores = np.exp(scores / (2. * (self.sigma ** 2)))
        # normalize the results
        # scores = normalize(scores)
        scores /= np.sum(scores)
        self.scores = np.asarray(scores).squeeze()
        # pick element to survive
        survivors = np.random.choice(a=len(self.particles), size=self.N, replace=True, p=self.scores)
        self.particles = list(map(lambda i: Particle(self.particles[i].state), survivors))
        self.scores = list(map(lambda i: self.scores[i], survivors))
        self.scores /= np.sum(self.scores)

    def draw_particles(self, img):
        for p in self.particles:
            img = cv2.circle(img, p.get_point(), 2, (255, 0, 0))
        return img

    def draw_tracking_window(self, img):
        weighted_u_sum = 0
        weighted_v_sum = 0
        for i in range(len(self.scores)):
            weighted_u_sum += self.scores[i] * self.particles[i].state[0]
            weighted_v_sum += self.scores[i] * self.particles[i].state[1]
        weighted_v_sum = int(weighted_v_sum)
        weighted_u_sum = int(weighted_u_sum)
        w_u = int(self.window.shape[0]/2)
        w_v = int(self.window.shape[1]/2)
        pt1 = (weighted_u_sum - w_u, weighted_v_sum - w_v)
        pt2 = (weighted_u_sum + w_u, weighted_v_sum + w_v)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0))
        cv2.circle(img, (weighted_u_sum, weighted_v_sum), np.var(self.scores), (0, 255, 0))
        return img


def normalize(img):
    mini = np.min(img)
    maxi = np.max(img)
    return np.vectorize(lambda x: (x - mini)/(maxi - mini))(img)
