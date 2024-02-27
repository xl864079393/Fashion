import torch

# Create a KNN model using pytorch and calculate it
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        m = X.size(0)
        n = self.X_train.size(0)

        xx = (X**2).sum(dim = 1, keepdim = True).expand(m, n)
        yy = (self.X_train**2).sum(dim = 1, keepdim = True).expand(n, m).t()

        dist = xx + yy - 2 * torch.mm(X, self.X_train.t())
        _, index = dist.topk(self.k, dim = 1, largest = False)
        pred = self.y_train[index]
        pred = pred.mode(dim = 1)[0]
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).float().mean().item()