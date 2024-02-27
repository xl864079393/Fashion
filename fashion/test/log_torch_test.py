from fashion.models.log_torch import LogisticRegressionModel
import torch

def log_torch_test(X_train, y_train, X_test, y_test, classes):
    input_dim = X_train.size(1)
    output_dim = len(classes)
    model = LogisticRegressionModel(input_dim, output_dim, device = "cuda")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    model.fit(X_train, y_train, criterion, optimizer, epochs = 1000)
    print(model.score(X_test, y_test))