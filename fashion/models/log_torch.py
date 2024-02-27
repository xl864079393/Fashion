# Build a logistic regression model to classify fashion images
import torch
from torch import nn
from torch.nn.modules.module import T


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, device="cpu"):
        self.device = device
        if device == "cuda":
            assert torch.cuda.is_available(), "Cuda is not available"
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, criterion, optimizer, epochs=100):
        if self.device == "cuda":
            self.to("cuda")
            X = X.to("cuda")
            y = y.to("cuda")

        for epoch in range(epochs):
            # Forward pass
            outputs = self(X)
            loss = criterion(outputs, y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch+1) % 100 == 0:
                print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
        return self

    def predict(self, X):
        return self(X)

    def score(self, X, y):
        if self.device == "cuda":
            X = X.to("cuda")
            y = y.to("cuda")
        y_pred = self.predict(X)
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y).float().mean().item()

