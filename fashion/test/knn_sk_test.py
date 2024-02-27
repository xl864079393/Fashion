from fashion.models.knn_sk import knn_sk
import time

def knn_sk_test(X_train, y_train, X_test, y_test, K=3, dim1=False):
    knc = knn_sk()
    knc.fit(X_train, y_train)
    start = time.time()
    print(knc.score(X_test, y_test))
    print("Time taken: ", time.time() - start)