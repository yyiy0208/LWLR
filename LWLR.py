from sklearn.linear_model import LinearRegression
import numpy as np


class LWLR(object):

    def __init__(self, k=0.1):
        self.k = k

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        return self

    def predict(self, X):
        result = []
        for example in np.array(X):
            prediction = self._predict_single(example)
            result.append(prediction)
        return np.array(result)

    def _predict_single(self, x):
        weight_vec = []
        for dis_vec in (self.X - x):
            dis = np.sum(np.square(dis_vec))
            weight = np.exp(-dis / (self.k * self.k * 2))
            weight_vec.append(weight)
        lnm = LinearRegression()
        lnm.fit(self.X, self.y, sample_weight=np.array(weight_vec))
        return lnm.predict([x])