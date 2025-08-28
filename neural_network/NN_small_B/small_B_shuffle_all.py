import numpy as np
import pandas as pd

# ****************************************
# data_id_shuffle_all.py
# 打乱数据集的顺序
# 输出打乱后的顺序
# ****************************************

# 各区域的网格单元的个数
num_each_region = [1016, 1034, 1009, 1028]
num_each_col = [1023, 2057, 3066, 4094]

# 设置训练集样本占比
percentage_train = 0.96

path_save = "C:\\project\\511\\neural_network\\NN_small_data\\"
name_data_R1 = 'small_B_data_S_R1.csv'
name_testset = 'small_B_test_S_all.csv'
name_shuffle = 'small_B_shuffle_id_S_all.csv'

# 样本个数
path_data_R1 = path_save + name_data_R1
data_NN_R1 = pd.read_csv(path_data_R1, header=None, encoding='utf-8').to_numpy()

shape_data = data_NN_R1.shape
print('成功读入数据集，大小: ', shape_data)
len_data = data_NN_R1.shape[0]
print('样本个数: ', len_data)

# **********

len_train = int(percentage_train * len_data)
len_test = len_data - len_train
print('训练集占比: ', percentage_train)
print('训练集样本个数: ', len_train)
print('测试集样本个数: ', len_test)

# shuffle_id是数组，[0, 1, 2, ...]
shuffle_id = [int(ii) for ii in range(len_data)]
# 打乱
np.random.shuffle(shuffle_id)
# 保存打乱后的
# 保存路径
path_shuffle = path_save + name_shuffle
# 先转化为DataFrame格式，再保存
df = pd.DataFrame(shuffle_id)
df.to_csv(path_shuffle, encoding='utf-8', index=False, header=False)
print('Shuffled training set id is successfully saved! path: ', path_shuffle)

# **********

col_data = 7
for i in range(4):
    col_data += num_each_region[i]

# **********

# shuffle_id的后len_test个数，是测试集
path_testset = path_save + name_testset
data_test = np.zeros((len_test, col_data))

for j in range(len_test):
    data_test[j][0:int(num_each_col[0])] = data_NN_R1[shuffle_id[len_train + j]][0:int(num_each_region[0] + 7)]
print(path_data_R1)

for i in range(3):
    name_data = 'small_B_data_S_R' + str(i + 2) + '.csv'
    path_data = path_save + name_data
    print(path_data)
    data_NN = pd.read_csv(path_data, header=None, encoding='utf-8').to_numpy()
    for j in range(len_test):
        data_test[j][int(num_each_col[i]):int(num_each_col[i + 1])] = data_NN[shuffle_id[len_train + j]][
                                                                      7:int(num_each_region[i + 1] + 7)]
df_test = pd.DataFrame(data_test)
df_test.to_csv(path_testset, encoding='utf-8', index=False, header=False)

print("end")
