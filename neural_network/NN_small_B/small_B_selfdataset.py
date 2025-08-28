import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# 各区域的网格单元的个数
num_each_region = [1016, 1034, 1009, 1028]
num_each_region_core = [112, 127, 115, 127]
num_each_col = [1023, 2057, 3066, 4094]


class SelfDataset_S(Dataset):
    def __init__(self, file_path, target_field_id):
        df = pd.read_csv(file_path, header=None, encoding='utf-8')

        arr_FM = df.iloc[:, 1:7].values
        arr_FM[:, 0] = arr_FM[:, 0] / 120 + 0.5
        arr_FM[:, 1] = arr_FM[:, 1] / 120 + 0.5
        arr_FM[:, 2] = arr_FM[:, 2] / 400 + 0.5
        arr_FM[:, 3] = arr_FM[:, 3] / 2 + 0.5
        arr_FM[:, 4] = arr_FM[:, 4] / 2 + 0.5
        arr_FM[:, 5] = arr_FM[:, 5] / 4 + 0.5

        num_this_region = num_each_region[int(target_field_id - 1)]
        arr_S = df.iloc[:, 7:int(num_this_region + 7)].values * 0.000000001

        self.X = torch.from_numpy(arr_FM)
        self.Y1 = torch.from_numpy(arr_S)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y1[index]


class SelfDataset_E(Dataset):
    def __init__(self, file_path, target_field_id):
        df = pd.read_csv(file_path, header=None, encoding='utf-8')

        arr_FM = df.iloc[:, 1:7].values
        arr_FM[:, 0] = arr_FM[:, 0] / 120 + 0.5
        arr_FM[:, 1] = arr_FM[:, 1] / 120 + 0.5
        arr_FM[:, 2] = arr_FM[:, 2] / 400 + 0.5
        arr_FM[:, 3] = arr_FM[:, 3] / 2 + 0.5
        arr_FM[:, 4] = arr_FM[:, 4] / 2 + 0.5
        arr_FM[:, 5] = arr_FM[:, 5] / 4 + 0.5

        num_this_region = num_each_region[int(target_field_id - 1)]
        arr_E = df.iloc[:, 7:int(6 * num_this_region + 7)].values * 1000

        self.X = torch.from_numpy(arr_FM)
        self.Y2 = torch.from_numpy(arr_E)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y2[index]


class SelfDataset_S_core(Dataset):
    def __init__(self, file_path, target_field_id):
        df = pd.read_csv(file_path, header=None, encoding='utf-8')

        arr_FM = df.iloc[:, 1:7].values
        arr_FM[:, 0] = arr_FM[:, 0] / 120 + 0.5
        arr_FM[:, 1] = arr_FM[:, 1] / 120 + 0.5
        arr_FM[:, 2] = arr_FM[:, 2] / 400 + 0.5
        arr_FM[:, 3] = arr_FM[:, 3] / 2 + 0.5
        arr_FM[:, 4] = arr_FM[:, 4] / 2 + 0.5
        arr_FM[:, 5] = arr_FM[:, 5] / 4 + 0.5

        num_this_region_core = num_each_region_core[int(target_field_id - 1)]
        arr_E = df.iloc[:, 7:int(num_this_region_core + 7)].values * 0.000000001

        self.X = torch.from_numpy(arr_FM)
        self.Y1 = torch.from_numpy(arr_E)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y1[index]


class SelfDataset_E_core(Dataset):
    def __init__(self, file_path, target_field_id):
        df = pd.read_csv(file_path, header=None, encoding='utf-8')

        arr_FM = df.iloc[:, 1:7].values
        arr_FM[:, 0] = arr_FM[:, 0] / 120 + 0.5
        arr_FM[:, 1] = arr_FM[:, 1] / 120 + 0.5
        arr_FM[:, 2] = arr_FM[:, 2] / 400 + 0.5
        arr_FM[:, 3] = arr_FM[:, 3] / 2 + 0.5
        arr_FM[:, 4] = arr_FM[:, 4] / 2 + 0.5
        arr_FM[:, 5] = arr_FM[:, 5] / 4 + 0.5

        num_this_region_core = num_each_region_core[int(target_field_id - 1)]
        arr_E = df.iloc[:, 7:int(6 * num_this_region_core + 7)].values * 1000

        self.X = torch.from_numpy(arr_FM)
        self.Y2 = torch.from_numpy(arr_E)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y2[index]

class SelfDataset_S_all(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path, header=None, encoding='utf-8')

        arr_FM = df.iloc[:, 1:7].values
        arr_FM[:, 0] = arr_FM[:, 0] / 120 + 0.5
        arr_FM[:, 1] = arr_FM[:, 1] / 120 + 0.5
        arr_FM[:, 2] = arr_FM[:, 2] / 400 + 0.5
        arr_FM[:, 3] = arr_FM[:, 3] / 2 + 0.5
        arr_FM[:, 4] = arr_FM[:, 4] / 2 + 0.5
        arr_FM[:, 5] = arr_FM[:, 5] / 4 + 0.5

        arr_S_R1 = df.iloc[:, 7:int(num_each_col[0])].values * 0.000000001
        arr_S_R2 = df.iloc[:, int(num_each_col[0]):int(num_each_col[1])].values * 0.000000001
        arr_S_R3 = df.iloc[:, int(num_each_col[1]):int(num_each_col[2])].values * 0.000000001
        arr_S_R4 = df.iloc[:, int(num_each_col[2]):int(num_each_col[3])].values * 0.000000001

        self.X = torch.from_numpy(arr_FM)
        self.Y11 = torch.from_numpy(arr_S_R1)
        self.Y12 = torch.from_numpy(arr_S_R2)
        self.Y13 = torch.from_numpy(arr_S_R3)
        self.Y14 = torch.from_numpy(arr_S_R4)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y11[index], self.Y12[index], self.Y13[index], self.Y14[index]