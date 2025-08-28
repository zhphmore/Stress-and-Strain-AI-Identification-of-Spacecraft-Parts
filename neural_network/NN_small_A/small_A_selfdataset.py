import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class SelfDataset(Dataset):
    def __init__(self, path_data):
        df = pd.read_csv(path_data,
                         usecols=["C1", "C2", "C3", "C4", "C5", "C6", "Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                         dtype={"C1": np.float32, "C2": np.float32, "C3": np.float32, "C4": np.float32,
                                "C5": np.float32, "C6": np.float32, "Fx": np.float32, "Fy": np.float32,
                                "Fz": np.float32, "Mx": np.float32, "My": np.float32, "Mz": np.float32},
                         header=0, encoding="utf-8")

        arr_clip = df.iloc[:, :6].values * 100000
        arr_FM = df.iloc[:, 6:12].values
        arr_FM[:, 0] = arr_FM[:, 0] / 120 + 0.5
        arr_FM[:, 1] = arr_FM[:, 1] / 120 + 0.5
        arr_FM[:, 2] = arr_FM[:, 2] / 400 + 0.5
        arr_FM[:, 3] = arr_FM[:, 3] / 2 + 0.5
        arr_FM[:, 4] = arr_FM[:, 4] / 2 + 0.5
        arr_FM[:, 5] = arr_FM[:, 5] / 4 + 0.5

        self.X = torch.from_numpy(arr_clip)
        self.Y = torch.from_numpy(arr_FM)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def revertData(test_X, outputs, test_Y):
    in_val = test_X.squeeze(0).numpy()
    in_val /= 100000

    # out_val = np.zeros(6)
    # real_val = np.zeros(6)

    out_val = outputs.cpu().squeeze(0).numpy()
    real_val = test_Y.cpu().squeeze(0).numpy()

    out_val -= 0.5
    real_val -= 0.5
    for k in range(6):
        if k == 0 or k == 1:
            out_val[k] *= 120
            real_val[k] *= 120
        elif k == 2:
            out_val[k] *= 400
            real_val[k] *= 400
        elif k == 3 or k == 4:
            out_val[k] *= 2
            real_val[k] *= 2
        elif k == 5:
            out_val[k] *= 4
            real_val[k] *= 4

    return in_val, out_val, real_val
