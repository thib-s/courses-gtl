from sklearn.mixture import GaussianMixture


class GaussianMixtureTransform(GaussianMixture):
    def transform(self, X):
        return self.predict(X).reshape(-1, 1)