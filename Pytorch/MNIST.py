import torch 
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


"""
Datasets
"""

training_data = datasets.FashionMNIST(
        root="data", 
        train=True, 
        download=True, 
        transform=ToTensor()
        )
test_data = datasets.FashionMNIST(
        root="data", 
        train=False, 
        download=True, 
        transform=ToTensor()
        )

#TODO - open local data files.

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

input_size= 28*28


"""
Network define
"""

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        # linear network 4 layers
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 512), 
                nn.ReLU(), 
                nn.Linear(512, 512), 
                nn.ReLU(), 
                nn.Linear(512, 10), 
                )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


 


"""
params
"""
alpha = 0.05
bs = 64
iterations = 10

model = NeuralNetwork()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(
        model.parameters(), 
        lr=alpha
        )

"""
Optimization Loop
"""


def train(
        dataloader, 
        model, 
        loss_fn, 
        optimizer):

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        #predictions
        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()


        if batch % 100 == 0: 
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(
        dataloader, 
        model, 
        loss_fn
        ):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader: 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")


for t in range(iterations): 
    print(f"Epoch {t+1}\n -----------")
    train(train_dataloader, model, loss, optim)
    test_loop(test_dataloader, model, loss)
print("Done!")

