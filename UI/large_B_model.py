import torch
import torch.nn as nn

# 各区域的网格单元的个数
num_each_region = [735, 457, 453, 705, 730, 455, 455, 703]
num_each_region_core = [125, 115, 117, 119, 126, 116, 118, 112]


class NN_S(torch.nn.Module):
    def __init__(self, target_field_id):
        super(NN_S, self).__init__()

        num_this_region = num_each_region[int(target_field_id - 1)]
        output_dim = int(num_this_region)

        self.linear1 = nn.Linear(6, 16)
        self.linear2 = nn.Linear(16, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, 512)
        self.linear6 = nn.Linear(512, 512)
        self.linear7 = nn.Linear(512, 512)
        self.linear8 = nn.Linear(512, 1024)
        self.linear9 = nn.Linear(1024, 1024)
        self.linear10 = nn.Linear(1024, 1024)
        self.linear11 = nn.Linear(1024, output_dim)

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
        nn.init.xavier_uniform_(self.linear8.weight)
        nn.init.xavier_uniform_(self.linear9.weight)
        nn.init.xavier_uniform_(self.linear10.weight)
        nn.init.xavier_uniform_(self.linear11.weight)

    def forward(self, x):
        x = x.to(torch.float32)
        h1 = self.linear1(x)
        h11 = self.leakyrelu1(h1)
        h2 = self.linear2(h11)
        h21 = self.leakyrelu1(h2)
        h3 = self.linear3(h21)
        h31 = self.leakyrelu1(h3)
        h4 = self.linear4(h31)
        h41 = self.leakyrelu1(h4)
        h5 = self.linear5(h41)
        h51 = self.leakyrelu1(h5)
        h6 = self.linear6(h51)
        h61 = self.leakyrelu1(h6)
        h7 = self.linear7(h61)
        h71 = self.leakyrelu1(h7)
        h8 = self.linear8(h71)
        h81 = self.leakyrelu1(h8)
        h9 = self.linear9(h81)
        h91 = self.leakyrelu1(h9)
        h10 = self.linear10(h91)
        h101 = self.leakyrelu1(h10)
        y = self.linear11(h101)

        return y


class NN_E(torch.nn.Module):
    def __init__(self, target_field_id):
        super(NN_E, self).__init__()

        num_this_region = num_each_region[int(target_field_id - 1)]
        output_dim = int(6 * num_this_region)

        self.linear1 = nn.Linear(6, 16)
        self.linear2 = nn.Linear(16, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, 512)
        self.linear6 = nn.Linear(512, 1024)
        self.linear7 = nn.Linear(1024, 2048)
        self.linear8 = nn.Linear(2048, 2048)
        self.linear9 = nn.Linear(2048, 2048)
        self.linear10 = nn.Linear(2048, 1024)
        self.linear11 = nn.Linear(1024, output_dim)

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
        nn.init.xavier_uniform_(self.linear8.weight)
        nn.init.xavier_uniform_(self.linear9.weight)
        nn.init.xavier_uniform_(self.linear10.weight)
        nn.init.xavier_uniform_(self.linear11.weight)

    def forward(self, x):
        x = x.to(torch.float32)
        h1 = self.linear1(x)
        h11 = self.leakyrelu1(h1)
        h2 = self.linear2(h11)
        h21 = self.leakyrelu1(h2)
        h3 = self.linear3(h21)
        h31 = self.leakyrelu1(h3)
        h4 = self.linear4(h31)
        h41 = self.leakyrelu1(h4)
        h5 = self.linear5(h41)
        h51 = self.leakyrelu1(h5)
        h6 = self.linear6(h51)
        h61 = self.leakyrelu1(h6)
        h7 = self.linear7(h61)
        h71 = self.leakyrelu1(h7)
        h8 = self.linear8(h71)
        h81 = self.leakyrelu1(h8)
        h9 = self.linear9(h81)
        h91 = self.leakyrelu1(h9)
        h10 = self.linear10(h91)
        h101 = self.leakyrelu1(h10)
        y = self.linear11(h101)

        return y


class NN_S_core(torch.nn.Module):
    def __init__(self, target_field_id):
        super(NN_S_core, self).__init__()

        num_this_region_core = num_each_region_core[int(target_field_id - 1)]
        output_dim = int(num_this_region_core)

        self.linear1 = nn.Linear(6, 16)
        self.linear2 = nn.Linear(16, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 512)
        self.linear5 = nn.Linear(512, 512)
        self.linear6 = nn.Linear(512, 1024)
        self.linear7 = nn.Linear(1024, 1024)
        self.linear8 = nn.Linear(1024, 1024)
        self.linear9 = nn.Linear(1024, 512)
        self.linear10 = nn.Linear(512, 256)
        self.linear11 = nn.Linear(256, output_dim)

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
        nn.init.xavier_uniform_(self.linear8.weight)
        nn.init.xavier_uniform_(self.linear9.weight)
        nn.init.xavier_uniform_(self.linear10.weight)
        nn.init.xavier_uniform_(self.linear11.weight)

    def forward(self, x):
        x = x.to(torch.float32)
        h1 = self.linear1(x)
        h11 = self.leakyrelu1(h1)
        h2 = self.linear2(h11)
        h21 = self.leakyrelu1(h2)
        h3 = self.linear3(h21)
        h31 = self.leakyrelu1(h3)
        h4 = self.linear4(h31)
        h41 = self.leakyrelu1(h4)
        h5 = self.linear5(h41)
        h51 = self.leakyrelu1(h5)
        h6 = self.linear6(h51)
        h61 = self.leakyrelu1(h6)
        h7 = self.linear7(h61)
        h71 = self.leakyrelu1(h7)
        h8 = self.linear8(h71)
        h81 = self.leakyrelu1(h8)
        h9 = self.linear9(h81)
        h91 = self.leakyrelu1(h9)
        h10 = self.linear10(h91)
        h101 = self.leakyrelu1(h10)
        y = self.linear11(h101)

        return y


class NN_E_core(torch.nn.Module):
    def __init__(self, target_field_id):
        super(NN_E_core, self).__init__()

        num_this_region_core = num_each_region_core[int(target_field_id - 1)]
        output_dim = int(6 * num_this_region_core)

        self.linear1 = nn.Linear(6, 16)
        self.linear2 = nn.Linear(16, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, 512)
        self.linear6 = nn.Linear(512, 1024)
        self.linear7 = nn.Linear(1024, 2048)
        self.linear8 = nn.Linear(2048, 2048)
        self.linear9 = nn.Linear(2048, 2048)
        self.linear10 = nn.Linear(2048, 1024)
        self.linear11 = nn.Linear(1024, output_dim)

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
        nn.init.xavier_uniform_(self.linear8.weight)
        nn.init.xavier_uniform_(self.linear9.weight)
        nn.init.xavier_uniform_(self.linear10.weight)
        nn.init.xavier_uniform_(self.linear11.weight)

    def forward(self, x):
        x = x.to(torch.float32)
        h1 = self.linear1(x)
        h11 = self.leakyrelu1(h1)
        h2 = self.linear2(h11)
        h21 = self.leakyrelu1(h2)
        h3 = self.linear3(h21)
        h31 = self.leakyrelu1(h3)
        h4 = self.linear4(h31)
        h41 = self.leakyrelu1(h4)
        h5 = self.linear5(h41)
        h51 = self.leakyrelu1(h5)
        h6 = self.linear6(h51)
        h61 = self.leakyrelu1(h6)
        h7 = self.linear7(h61)
        h71 = self.leakyrelu1(h7)
        h8 = self.linear8(h71)
        h81 = self.leakyrelu1(h8)
        h9 = self.linear9(h81)
        h91 = self.leakyrelu1(h9)
        h10 = self.linear10(h91)
        h101 = self.leakyrelu1(h10)
        y = self.linear11(h101)

        return y
