import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torchsummary import summary
from torchviz import make_dot
from torchvision import datasets, transforms


class SingleLayerCNN(nn.Module):
    def __init__(self):
        super(SingleLayerCNN, self).__init__()
        self.conv = nn.Conv2d(1, 2, 16)
        self.conn = nn.Linear(338, 10)
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.conn(x)
        return x

torch.manual_seed(42)
net = SingleLayerCNN()

summary(net, (1,28,28), device="cpu")

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
test_acc_hist = []
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
        # loss_hist.append(loss_val.item())
        iter_counter += 1
        if iter_counter % 50 == 0:
            with torch.no_grad():
                test_loss = 0
                test_correct = 0
                test_less_ctr = 0
                for data, targets in test_loader:
                    net_out = net(data)
                    test_loss += loss(net_out, targets)
                    predicted = net_out.argmax()
                    test_correct += predicted==targets
                    test_less_ctr += 1
                    if test_less_ctr >= 50:
                        break
                test_loss_hist.append(test_loss / 50)
                test_acc_hist.append(test_correct / 50)

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

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot([i*50 for i in range(len(test_loss_hist))], test_loss_hist)
plt.title("MNIST Loss Curve for CNN")
plt.legend(["Test Loss"])
plt.xlabel("Items Seen")
plt.ylabel("Loss")
plt.show()

print(test_acc_hist)
# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot([i*50 for i in range(len(test_acc_hist))], test_acc_hist)
plt.title("MNIST Accuracy Curve for CNN")
plt.legend(["Test Accuracy"])
plt.xlabel("Items Seen")
plt.ylabel("Accuracy")
plt.show()

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_flops=True) as prof:
    with torch.no_grad():
        net.eval()
        data, targets = next(iter(test_loader))
        net_out = net(data)
        predicted = net_out.argmax()
print("Profiler results for one inference example:")
print(prof.key_averages().table(sort_by="cpu_memory_usage"))

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_flops=True) as prof:
    train_batch = iter(train_loader)
    print(train_batch)
    for data, targets in train_batch:
        net.train()
        net_out = net(data)
        loss_val = loss(net_out,targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        break
print("Profiler results for one training example:")
print(prof.key_averages().table(sort_by="cpu_memory_usage"))

yhat = net(data)
make_dot(yhat, params=dict(list(net.named_parameters()))).render("cnn_torchviz", format="png")