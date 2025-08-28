import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import large_B_model
import large_B_selfdataset

start_time = time.time()

# **********区域**********
# 注意程序运行前，请修改这里
# 准备测试的区域
target_field_id = 1

# **********模型参数保存的位置**********
# 注意程序运行前，请修改这里
# name_pth是程序输出的训练好的模型的文件名称
path_save = "C:\\project\\511\\neural_network\\NN_large_data\\"
name_pth = 'large_B_S_core_top5_R' + str(target_field_id) + '_1.pth'
name_testset = 'large_B_test_S_core_top5_R' + str(target_field_id) + '.csv'

# 各区域的网格单元的个数
num_each_region_core = [125, 115, 117, 119, 126, 116, 118, 112]
num_this_region = num_each_region_core[int(target_field_id - 1)]

batch_size = 1

# **********读取数据**********
# 读取测试集
path_testset = path_save + name_testset
test_df = large_B_selfdataset.SelfDataset_S_core(path_testset, target_field_id)
test_num = len(test_df)
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=True)
# test_num = 20

# **********加载模型**********
# 设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 注意
# 记得修改模型参数保存的位置
path_model = path_save + name_pth

loaded_model = large_B_model.NN_S_core(target_field_id)
loaded_model.load_state_dict(torch.load(path_model, weights_only=True))
print("model loaded successfully")
# 损失函数
criterion_1 = nn.L1Loss(reduction='mean')
# criterion_2 = nn.MSELoss(reduction='sum')

# **********测试预测准确率**********
loaded_model.eval()

ans = np.zeros((test_num, int(num_this_region)))
ans_pct = np.zeros((test_num, int(num_this_region)))
ans_each = np.zeros(int(num_this_region))
ans_each_pct = np.zeros(int(num_this_region))
max_mis = 0
max_pct = 0
ave_mis = 0
ave_pct = 0

total_test_loss = 0.0
with torch.no_grad():
    for step, data in enumerate(test_loader):
        if step < test_num:
            test_X, test_Y1 = data

            outputs = loaded_model(test_X)
            test_loss = criterion_1(outputs, test_Y1)
            total_test_loss = total_test_loss + test_loss

            for j in range(int(num_this_region)):
                ans_each[j] = abs(outputs[0][j] - test_Y1[0][j]) / 1000
                ans_each_pct[j] = abs(ans_each[j] / (test_Y1[0][j] / 1000))

            ans[step] = ans_each
            ans_pct[step] = ans_each_pct

            max_mis = max(max_mis, np.max(ans_each))
            max_pct = max(max_pct, np.max(ans_each_pct))
            ave_mis += np.mean(ans_each)
            ave_pct += np.mean(ans_each_pct)

            print("testing: {} of {}".format(step, test_num))

avg_test_loss = total_test_loss / test_num

ave_mis /= test_num
ave_pct /= test_num

print('max mistake: ', max_mis, ave_mis)
print('max percentage: ', max_pct, ave_pct)

df_ans = pd.DataFrame(ans)
path_4 = path_save + 'large_B_ans_E_core_R' + str(target_field_id) + '.csv'
df_ans.to_csv(path_4, header=False, index=False, encoding='utf-8')

print("total loss: ", total_test_loss)
print("average loss: ", avg_test_loss)
