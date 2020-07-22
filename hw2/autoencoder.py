import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from sklearn.decomposition import PCA
import sys

E = [[] for _ in range(6)]

def read_data(filename):
    X = []
    f = open(filename, 'r')
    for line in f:
        X.append(list(map(float, line.split())))
    X = np.array(X)[:, -256:]
    return X

class autoencoder:
    def __init__(self, hidden_size = 2, train_data = None, test_data = None, tied = False):
        self.train_X = train_data
        self.test_X = test_data
        self.AE = self.ae(hidden_size, tied).cuda()

    class ae(nn.Module):
        def __init__(self, hidden_size, tied = False):
            super().__init__()
            self.tied = tied
            U = (6 / (1 + 256 + hidden_size)) ** 0.5
            self.Tanh = nn.Tanh()
            if not tied:
                self.encoder = nn.Linear(256, hidden_size)
                self.decoder = nn.Linear(hidden_size, 256)
                self.encoder.bias.data.uniform_(-U, U)
                self.encoder.weight.data.uniform_(-U, U)
                self.decoder.bias.data.uniform_(-U, U)
                self.decoder.weight.data.uniform_(-U, U)
            else:
                self.param = nn.Parameter(torch.FloatTensor(np.random.uniform(-U, U, (hidden_size, 256))))
                self.encoder_bias = torch.FloatTensor(hidden_size).uniform_(-U, U).cuda()
                self.decoder_bias = torch.FloatTensor(256).uniform_(-U, U).cuda()


        def forward(self, x):
            if not self.tied:
                x = self.encoder(x)
                x = self.Tanh(x)
                x = self.decoder(x)
            else:
                x = F.linear(x, self.param, bias = self.encoder_bias)
                x = self.Tanh(x)
                x = F.linear(x, self.param.t(), bias = self.decoder_bias)
            return x

    def train(self, epoch = 5000, lr = 0.1, ind = 0):
        self.AE.train()
        global E
        Loss = nn.MSELoss()
        optimizer = torch.optim.SGD(self.AE.parameters(), lr = lr)
        for step in range(epoch):
            print('\r', end='')
            print(f'epoch: {step}', end='', flush=True)
            optimizer.zero_grad()
            output = self.AE(torch.FloatTensor(self.train_X).cuda())
            loss = Loss(output, torch.FloatTensor(self.train_X).cuda())
            if step == epoch - 1:
                print()
                print(f'train loss: {loss.item()}')
                E[ind].append(loss.item())
            loss.backward()
            optimizer.step()

    def test(self, ind = 0):
        global E
        self.AE.eval()
        Loss = nn.MSELoss()
        with torch.no_grad():
            output = self.AE(torch.FloatTensor(self.test_X).cuda())
            loss = Loss(output, torch.FloatTensor(self.test_X).cuda())
            print(f'test loss: {loss.item()}')
            E[ind].append(loss.item())

def Pca(hidden_size, train_data, test_data, ind = 0):
    global E
    clf = PCA(n_components = hidden_size)
    latents = clf.fit_transform(train_data)
    m = np.mean(train_data, axis = 0)
    w = clf.components_
    inv = (w.T@w@(train_data - m).T).T + m
    Loss = nn.MSELoss()
    loss = Loss(torch.Tensor(inv), torch.Tensor(train_data))
    print(f'train loss: {loss.item()}')
    E[ind].append(loss.item())
    m = np.mean(test_data, axis = 0)
    inv = (w.T@w@(test_data - m).T).T + m
    loss = Loss(torch.Tensor(inv), torch.Tensor(test_data))
    print(f'test loss: {loss.item()}')
    E[ind + 1].append(loss.item())

    

def visualize(x, ind, x_axis, y_axis, title, filename = ''):
    global E
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    color = ['g', 'b', 'r']
    for c, i in enumerate(ind):
        plt.plot(x, E[i], f'-{color[c]}o', label = f'Problem {11 + i}')
    plt.legend()
    if len(filename):
        plt.savefig(filename)
    plt.clf()

trainX = read_data('zip.train')
testX = read_data('zip.test')
lst = [i for i in range(1, 8)]

for d in lst:
    model = autoencoder(2**d, trainX, testX)
    model.train(ind = 0)
    model.test(ind = 1)

visualize(lst, [0], r'$log_2{\tilde{d}}$', r'$E_{in}$', 'Problem 11', 'p11.png')
visualize(lst, [1], r'$log_2{\tilde{d}}$', r'$E_{out}$', 'Problem 12', 'p12.png')

for d in lst:
    model = autoencoder(2**d, trainX, testX, tied = True)
    model.train(ind = 2)
    model.test(ind = 3)

visualize(lst, [0, 2], r'$log_2{\tilde{d}}$', r'$E_{in}$', 'Problem 13', 'p13.png')
visualize(lst, [1, 3], r'$log_2{\tilde{d}}$', r'$E_{out}$', 'Problem 14', 'p14.png')

for d in lst:
    Pca(2**d, trainX, testX, ind = 4)

visualize(lst, [2, 4], r'$log_2{\tilde{d}}$', r'$E_{in}$', 'Problem 15', 'p15.png')
visualize(lst, [3, 5], r'$log_2{\tilde{d}}$', r'$E_{out}$', 'Problem 16', 'p16.png')
