import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score


class KernelDensityNB:
    def __init__(self, bandwidth):
        self.kernel = "gaussian"
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self._classes = np.unique(y)
        self._priors = np.zeros(self._classes.size, dtype=np.float64)
        self._kdes = np.zeros(self._classes.size, dtype=KernelDensity)
        for idx, c in enumerate(self._classes):
            X_of_class = X[y == c]
            self._priors[idx] = np.log(X_of_class.shape[0] / X.shape[0])
            self._kdes[idx] = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(X_of_class)

    def predict(self, X):
        all_scores = [self._priors[idx] + self._kdes[idx].score_samples(X)
                      for idx, _ in enumerate(self._classes)]

        return [self._classes[best_class] for best_class in np.argmax(all_scores, axis=0)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
