import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle

torch.manual_seed(42)

save_path = '/Users/samanthacoury/Desktop/snn'

# adapt from https://github.com/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb
# Define Network
class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# dataloader arguments
batch_size = 128
data_path='/tmp/data/mnist'

dtype = torch.float
device = torch.device("cpu")
# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
# drop_last switched to False to keep all samples
test_loader_eval = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

num_inputs = 28*28
num_hidden_units = [250, 500, 750, 1000]
# num_hidden_units = np.round(np.logspace(np.log10(2), np.log10(100), num=10)).astype(int)
num_outputs = 10
N_trials = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

final_eval = np.zeros((N_trials, len(num_hidden_units)))


for i, num_hidden in enumerate(num_hidden_units):
    if num_hidden == 0:
        continue
    print(f"Hidden units: {num_hidden}")
    for trial in range(N_trials):
        print(f"Trial: {trial}")
        net = Net(num_inputs, num_hidden, num_outputs, beta).to(device)

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

        # train for one epoch
        num_epochs = 1
        loss_hist = []
        test_loss_hist = []
        counter = 0

        # Outer training loop
        for epoch in range(num_epochs):
            iter_counter = 0
            train_batch = iter(train_loader)

            # Minibatch training loop
            for data, targets in train_batch:
                data = data.to(device)
                targets = targets.to(device)

                # forward pass
                net.train()
                spk_rec, mem_rec = net(data.view(batch_size, -1))

                # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                # Test set
                with torch.no_grad():
                    net.eval()
                    test_data, test_targets = next(iter(test_loader))
                    test_data = test_data.to(device)
                    test_targets = test_targets.to(device)

                    # Test set forward pass
                    test_spk, test_mem = net(test_data.view(batch_size, -1))

                    # Test set loss
                    test_loss = torch.zeros((1), dtype=dtype, device=device)
                    for step in range(num_steps):
                        test_loss += loss(test_mem[step], test_targets)
                    test_loss_hist.append(test_loss.item())
                    counter += 1
                    iter_counter +=1

        # Plot Loss
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        plt.plot(loss_hist)
        plt.plot(test_loss_hist)
        plt.title("Loss Curves for {} Hidden Units".format(num_hidden))
        plt.legend(["Train Loss", "Test Loss"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(f"{save_path}/loss_{num_hidden}_{trial}.png")
        plt.close()

        total = 0
        correct = 0

        with torch.no_grad():
            net.eval()
            for data, targets in test_loader_eval:
                data = data.to(device)
                targets = targets.to(device)

                # forward pass
                test_spk, _ = net(data.view(data.size(0), -1))

                # calculate total accuracy
                _, predicted = test_spk.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"Total correctly classified test set images: {correct}/{total}")
        print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
        final_eval[trial, i] = correct / total


# Plot final evaluation
plt.figure()
plt.errorbar(num_hidden_units, final_eval.mean(axis=0)*100, final_eval.std(axis=0)*100)
plt.title("Accuracy vs. Number of Hidden Units Over {} Trials".format(N_trials))
plt.xlabel("Number of Hidden Units")
plt.ylabel("Accuracy")
plt.savefig(f"{save_path}/hidden/final_eval.png")
plt.show()

with open(f"{save_path}/hidden/final_eval.npy", "wb") as f:
    pickle.dump(final_eval, f)


