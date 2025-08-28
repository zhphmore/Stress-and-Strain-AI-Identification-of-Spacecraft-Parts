import numpy as np
import pandas as pd

# ****************************************
# data_id_shuffle.py
# 打乱数据集的顺序
# 输出打乱后的顺序
# ****************************************

# **********区域**********
# 注意程序运行前，请修改这里
# 准备制作哪个区域的训练和测试用数据集
target_field_id = 2

# 设置训练集样本占比
percentage_train = 0.96

path_save = "C:\\project\\511\\neural_network\\NN_large_data\\"
name_data = 'large_B_data_S_R' + str(target_field_id) + '.csv'
name_trainset = 'large_B_train_S_R' + str(target_field_id) + '.csv'
name_testset = 'large_B_test_S_R' + str(target_field_id) + '.csv'
name_shuffle = 'large_B_shuffle_id_S_R' + str(target_field_id) + '.csv'
# name_data = 'large_B_data_E_R' + str(target_field_id) + '.csv'
# name_trainset = 'large_B_train_E_R' + str(target_field_id) + '.csv'
# name_testset = 'large_B_test_E_R' + str(target_field_id) + '.csv'
# name_shuffle = 'large_B_shuffle_id_E_R' + str(target_field_id) + '.csv'

# 样本个数
path_data = path_save + name_data
data_NN = pd.read_csv(path_data, header=None, encoding='utf-8').to_numpy()
shape_data = data_NN.shape
print('成功读入数据集，大小: ', shape_data)
len_data = data_NN.shape[0]
print('样本个数: ', len_data)
col_data = data_NN.shape[1]

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

# shuffle_id的前len_train个数，是训练集
path_trainset = path_save + name_trainset
data_train = np.zeros((len_train, col_data))
for i in range(len_train):
    data_train[i] = data_NN[shuffle_id[i]]
df_train = pd.DataFrame(data_train)
df_train.to_csv(path_trainset, encoding='utf-8', index=False, header=False)

# shuffle_id的后len_test个数，是测试集
path_testset = path_save + name_testset
data_test = np.zeros((len_test, col_data))
for i in range(len_test):
    data_test[i] = data_NN[shuffle_id[len_test + i]]
df_test = pd.DataFrame(data_test)
df_test.to_csv(path_testset, encoding='utf-8', index=False, header=False)

print("end")
