import numpy as np


class RSVLinearRegression:
    def __init__(self):
        self._w = None

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w):
        """Sets the w parameters, resetting any previous fitting"""
        assert len(w) >= 1, "w must have length >= 1"
        assert isinstance(w[0], np.float32) or isinstance(w[0], np.float64), \
            "w must a float64 or float32: type(w[0]) == " + str(type(w[0]))
        self._w = w

    def _sanitize_X(self, X):
        assert len(X) >= 1, "There must be at least one data point: len(X) ==" + str(len(X))
        assert len(X[0]), "x vector must have length of at least 1: len(X[0]) ==" + str(len(X[0]))
        if self.w is not None:  # +1 because bias is added internally
            assert len(X[0]) + 1 == len(self.w), \
                "length of x vectors must be the same as w parameters: len(w), len(X[0]) == " + \
                str(len(self.w)) + ", " + str(len(X[0]))

    def _sanitize_X_y(self, X, y):
        self._sanitize_X(X)
        assert len(y) == len(X), "y must be the same length as X"

    def fit(self, X, y):
        """Does a linear regression fit of the data X and target y, returning w"""
        self._w = None  # Clearing previous fit
        self._sanitize_X_y(X, y)

        # Adding bias to the model by adding an element 1 to all x vectors
        X_biased = [np.append([1], x) for x in X]

        # Solving X^TXw = X^Ty, in Ax=B form, A=X^TX and B=X^Ty
        self.w = np.linalg.solve(np.dot(np.transpose(X_biased), X_biased), np.dot(np.transpose(X_biased), y))
        return self.w

    def predict(self, X):
        assert self.w is not None, "fit() has to be called, or w set before predict() is called"
        self._sanitize_X(X)
        # Notice that bias is added before computing, since w will have the extra element w0
        return [np.dot(self.w, np.append([1], x)) for x in X]
