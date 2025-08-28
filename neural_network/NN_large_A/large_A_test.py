import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import large_A_selfdataset
import large_A_model

start_time = time.time()

# **********模型参数保存的位置**********
# 注意程序运行前，请修改这里
# name_pth是程序输出的训练好的模型的文件名称i
path_save = "C:\\project\\511\\neural_network\\NN_large_data\\"
name_pth = 'large_A_2.pth'
name_testset = 'large_A_test.csv'

batch_size = 1

# **********读取数据**********
# 读取测试集
path_testset = path_save + name_testset
test_df = large_A_selfdataset.SelfDataset(path_testset)
test_num = len(test_df)
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=True)
# test_num = 1

# **********加载模型**********
# 设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 注意
# 记得修改模型参数保存的位置
path_model = path_save + name_pth
# loaded_model = torch.load(path_model, map_location='cpu')
loaded_model = large_A_model.NN_FM()
loaded_model.load_state_dict(torch.load(path_model, map_location='cpu'))
print("model loaded successfully")
# 损失函数
criterion_1 = nn.L1Loss(reduction='mean')
# criterion_2 = nn.MSELoss(reduction='sum')

# **********测试预测准确率**********
loaded_model.eval()

mis_each = np.zeros(6)
mis_avg = np.zeros(6)
mis_max = np.zeros(6)
mis_pct = np.zeros(6)
cnt_num = np.zeros(6)
for k in range(6):
    cnt_num[k] = test_num

ans = np.zeros((test_num, 26))
ans_each = np.zeros(26)

total_test_loss = 0.0
with torch.no_grad():
    for step, data in enumerate(test_loader):
        if step < test_num:
            test_X, test_Y = data
            # if torch.cuda.is_available():
            #     test_X = test_X.cuda()
            #     test_Y = test_Y.cuda()

            outputs = loaded_model(test_X)
            test_loss = criterion_1(outputs, test_Y)
            total_test_loss = total_test_loss + test_loss

            in_val, out_val, real_val = large_A_selfdataset.revertData(test_X.cpu(), outputs.cpu(), test_Y.cpu())

            # print("input clip E: ", test_X)
            print("model output F M: ", out_val)
            print("real F M: ", real_val)
            # print("test loss: ", test_loss_1, test_loss_2, test_loss_3)

            ans_each[0:8] = in_val
            for k in range(6):
                mis_each[k] = abs(out_val[k] - real_val[k])
                if mis_each[k] > mis_max[k]:
                    mis_max[k] = mis_each[k]
                if real_val[k] == 0 or out_val[k] == 0:
                    cnt_num[k] -= 1
                else:
                    mis_avg[k] += mis_each[k]
                    mis_pct[k] += abs(mis_each[k] / out_val[k])
                ans_each[3 * k + 8] = out_val[k]
                ans_each[3 * k + 9] = real_val[k]
                ans_each[3 * k + 10] = abs(mis_each[k] / out_val[k])
            ans[step] = ans_each

            print("testing: {} of {}".format(step, test_num))

avg_test_loss = total_test_loss / test_num

for k in range(6):
    mis_pct[k] /= cnt_num[k]
    mis_avg[k] /= cnt_num[k]

df_mis_pct = pd.DataFrame(mis_pct.T)
path_1 = path_save + 'large_A_mis_pct.csv'
df_mis_pct.to_csv(path_1, header=False, index=False, encoding='utf-8')
df_mis_avg = pd.DataFrame(mis_avg.T)
path_2 = path_save + 'large_A_mis_avg.csv'
df_mis_avg.to_csv(path_2, header=False, index=False, encoding='utf-8')
df_mis_max = pd.DataFrame(mis_max.T)
path_3 = path_save + 'large_A_mis_max.csv'
df_mis_max.to_csv(path_3, header=False, index=False, encoding='utf-8')
df_ans = pd.DataFrame(ans)
path_4 = path_save + 'large_A_ans.csv'
df_ans.to_csv(path_4, header=False, index=False, encoding='utf-8')

print("total loss: ", total_test_loss)
print("average loss: ", avg_test_loss)
print("mistake percentage: ", mis_pct)
print("mistake average: ", mis_avg)
print("mistake max: ", mis_max)
