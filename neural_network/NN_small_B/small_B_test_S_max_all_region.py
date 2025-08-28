import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import small_B_model
import small_B_selfdataset

start_time = time.time()

# **********模型参数保存的位置**********
# 注意程序运行前，请修改这里
# name_pth是程序输出的训练好的模型的文件名称
path_save = "C:\\project\\511\\neural_network\\NN_small_data\\"
name_pth_R1 = 'small_B_S_R1_1.pth'
name_pth_R2 = 'small_B_S_R2_1.pth'
name_pth_R3 = 'small_B_S_R3_1.pth'
name_pth_R4 = 'small_B_S_R4_1.pth'
name_testset_all = 'small_B_test_S_all.csv'

# 各区域的网格单元的个数
num_each_region = [1016, 1034, 1009, 1028]

batch_size = 1

# **********读取数据**********
# 读取测试集
path_testset = path_save + name_testset_all
test_df = small_B_selfdataset.SelfDataset_S_all(path_testset)
test_num = len(test_df)
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=True)
# test_num = 20

# **********加载模型**********
# 设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 注意
# 记得修改模型参数保存的位置
path_model_R1 = path_save + name_pth_R1
path_model_R2 = path_save + name_pth_R2
path_model_R3 = path_save + name_pth_R3
path_model_R4 = path_save + name_pth_R4

loaded_model_R1 = small_B_model.NN_S(1)
loaded_model_R2 = small_B_model.NN_S(2)
loaded_model_R3 = small_B_model.NN_S(3)
loaded_model_R4 = small_B_model.NN_S(4)

loaded_model_R1.load_state_dict(torch.load(path_model_R1, weights_only=True))
loaded_model_R2.load_state_dict(torch.load(path_model_R2, weights_only=True))
loaded_model_R3.load_state_dict(torch.load(path_model_R3, weights_only=True))
loaded_model_R4.load_state_dict(torch.load(path_model_R4, weights_only=True))

print("model loaded successfully")
# 损失函数
criterion_1 = nn.L1Loss(reduction='mean')
# criterion_2 = nn.MSELoss(reduction='sum')

# **********测试预测准确率**********
loaded_model_R1.eval()
loaded_model_R2.eval()
loaded_model_R3.eval()
loaded_model_R4.eval()

ans = np.zeros((test_num, 4))
ans_each = np.zeros(4)
max_mis = 0
max_pct = 0
ave_mis = 0
ave_pct = 0

total_test_loss = 0.0
with torch.no_grad():
    for step, data in enumerate(test_loader):
        if step < test_num:
            test_X, test_Y1, test_Y2, test_Y3, test_Y4 = data

            outputs_R1 = loaded_model_R1(test_X)
            outputs_R2 = loaded_model_R2(test_X)
            outputs_R3 = loaded_model_R3(test_X)
            outputs_R4 = loaded_model_R4(test_X)

            # test_loss = criterion_1(outputs, test_Y1)
            # total_test_loss = total_test_loss + test_loss

            ans_each[0] = max(outputs_R1.max(), outputs_R2.max(), outputs_R3.max(), outputs_R4.max()) * 500
            ans_each[1] = max(test_Y1.max(), test_Y2.max(), test_Y3.max(), test_Y4.max()) * 500
            # id_arg = outputs.argmax()
            # ans_each[1] = test_Y1[0][id_arg] * 500
            ans_each[2] = abs(ans_each[0] - ans_each[1])
            ans_each[3] = abs(ans_each[2] / ans_each[0])

            ans[step] = ans_each
            max_mis = max(max_mis, ans_each[2])
            max_pct = max(max_pct, ans_each[3])

            print("testing: {} of {}".format(step, test_num))

avg_test_loss = total_test_loss / test_num

for i in range(test_num):
    ave_mis += ans[i][2]
    ave_pct += ans[i][3]
ave_mis /= test_num
ave_pct /= test_num

print('max mistake: ', max_mis, ave_mis)
print('max percentage: ', max_pct, ave_pct)

df_ans = pd.DataFrame(ans)
path_4 = path_save + 'small_B_ans_S_R_all.csv'
df_ans.to_csv(path_4, header=False, index=False, encoding='utf-8')

print("total loss: ", total_test_loss)
print("average loss: ", avg_test_loss)
