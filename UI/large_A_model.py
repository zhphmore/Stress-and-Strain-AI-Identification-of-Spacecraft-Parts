import torch
import torch.nn as nn


class NN_FM(nn.Module):
    def __init__(self):
        super(NN_FM, self).__init__()

        self.linear1 = nn.Linear(8, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 128)
        self.linear6 = nn.Linear(128, 32)
        self.linear7 = nn.Linear(32, 6)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.linear4.weight)
        nn.init.xavier_uniform_(self.linear5.weight)
        nn.init.xavier_uniform_(self.linear6.weight)
        nn.init.xavier_uniform_(self.linear7.weight)

    def forward(self, x):
        x = x.to(torch.float32)
        h1 = self.linear1(x)
        h11 = self.tanh(h1)
        h2 = self.linear2(h11)
        h21 = self.tanh(h2)
        h3 = self.linear3(h21)
        h31 = self.leakyrelu2(h3)
        h4 = self.linear4(h31)
        h41 = self.leakyrelu1(h4)
        h5 = self.linear5(h41)
        h51 = self.relu(h5)
        h6 = self.linear6(h51)
        h71 = self.relu(h6)
        y = self.linear7(h71)

        return y
