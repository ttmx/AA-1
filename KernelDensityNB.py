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
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = np.zeros(self._classes.size, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            [posterior] = self._kdes[idx].score_samples([x])
            posteriors[idx] = self._priors[idx] + posterior

        return self._classes[np.argmax(posteriors)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
