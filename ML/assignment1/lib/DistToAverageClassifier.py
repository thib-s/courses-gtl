from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class DistToAverageClassifier(BaseEstimator, ClassifierMixin):
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
        df = pd.DataFrame(X)
        df['label'] = y
        self.mean_df = df.groupby(['label']).mean()
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        dist = []
        for category in range(len(self.mean_df)):
            dist.append(mean_squared_error(x, self.mean_df.values[
                category]))  # np.sum(np.ndarray.flatten(np.subtract(np.square(arr),np.square(mean_df.values[category])))))
        return np.argmin(dist)

    def predict(self, X, y=None):
        try:
            getattr(self, "mean_df")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return [self._meaning(x) for x in X]

    def score(self, X, y=None, **kwargs):
        # counts number of values bigger than mean
        return accuracy_score(self.predict(X),y)
