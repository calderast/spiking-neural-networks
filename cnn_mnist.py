import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SingleLayerCNN(nn.Module):
    def __init__(self):
        super(SingleLayerCNN, self).__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.conn = nn.Linear(1352, 10)
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.conn(x)
        return x
        # return F.log_softmax(x, dim=1)

torch.manual_seed(42)
net = SingleLayerCNN()

# begin adapted from https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html#accuracy-metric
batch_size = 1
data_path='/tmp/data/mnist'
dtype = torch.float
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
data, targets = next(iter(train_loader))
net_out = net(data)

num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)
    for data, targets in train_batch:
        net.train()
        net_out = net(data)
        loss_val = loss(net_out,targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        loss_hist.append(loss_val.item())

total = 0
correct = 0
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    net_out = net(data)
    predicted = net_out.argmax()
    total += targets.size(0)
    correct += (predicted == targets).sum().item()
print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
# end adapted

