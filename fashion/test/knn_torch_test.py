from fashion.models.knn_torch import KNN
import time
import torch

def knn_torch_test(X_train, y_train, X_test, y_test, K=3):
    knn = KNN(K)
    knn.fit(X_train, y_train)
    start = time.time()
    print(knn.score(X_test, y_test))
    print("Time taken: ", time.time() - start)