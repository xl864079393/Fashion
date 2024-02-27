import torch
import time

from utils.Loading import Loading
from models.knn_torch import KNN
import numpy as np
from test.log_torch_test import log_torch_test
from test.knn_sk_test import knn_sk_test
from test.knn_torch_test import knn_torch_test

def load_data():
    loading = Loading()
    trainloader, testloader, classes = loading.load_data()
    return trainloader, testloader, classes

def preprocess_data(trainloader, testloader):
    X_train, y_train = torch.cat([x for x, _ in trainloader]), torch.cat([y for _, y in trainloader])
    X_test, y_test = torch.cat([x for x, _ in testloader]), torch.cat([y for _, y in testloader])
    return X_train, y_train, X_test, y_test

def main():
    trainloader, testloader, classes = load_data()
    X_train, y_train, X_test, y_test = preprocess_data(trainloader, testloader)




if __name__ == "__main__":
    main()