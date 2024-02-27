import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

## load fashion mnist dataset
class Loading:
    def __init__(self, batch_size=4, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    def load_data(self):
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                               download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=self.num_workers)

        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

        return trainloader, testloader, classes
