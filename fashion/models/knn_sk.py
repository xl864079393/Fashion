import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class knn_sk:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y, dim1 = False):
        if dim1:
            # make the X 1D
            X = X.reshape(-1, 1)
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        knc = KNeighborsClassifier(n_neighbors=self.k)
        knc.fit(self.X_train, self.y_train)
        return knc.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)