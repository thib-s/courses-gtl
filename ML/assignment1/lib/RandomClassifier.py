from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import accuracy_score


class RandomClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self):
        """
        Called when initializing the classifier
        """
        pass

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.trained = True
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return True if x >= self.trained else False

    def predict(self, X, y=None):
        return np.random.randint(10, size=len(X))

    def score(self, X, y=None, **kwargs):
        # counts number of values bigger than mean
        return accuracy_score(self.predict(X), y)
