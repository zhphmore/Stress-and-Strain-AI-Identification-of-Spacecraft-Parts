import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import large_B_model
import large_B_selfdataset

start_time = time.time()

# **********区域**********
# 注意程序运行前，请修改这里
# 准备训练的区域
target_field_id = 1

# **********模型参数保存的位置**********
# 注意程序运行前，请修改这里
# name_pth是程序输出的训练好的模型的文件名称
path_save = "C:\\project\\511\\neural_network\\NN_large_data\\"
name_pth = 'large_B_E_R' + str(target_field_id) + '_1.pth'
name_trainset = 'large_B_train_E_R' + str(target_field_id) + '.csv'
name_testset = 'large_B_test_E_R' + str(target_field_id) + '.csv'
name_refer = 'large_B_E_R' + str(target_field_id) + '_1.pth'
flag_data_refer = True
# flag_data_refer = False

batch_size = 64
epochs = 1000
learning_rate = 0.001

# **********读取数据**********
# 读取训练集
path_trainset = path_save + name_trainset
train_df = large_B_selfdataset.SelfDataset_E(path_trainset, target_field_id)
train_num = len(train_df)
train_loader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
# 读取测试集
path_testset = path_save + name_testset
test_df = large_B_selfdataset.SelfDataset_E(path_testset, target_field_id)
test_num = len(test_df)
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=True)

# **********模型训练设置**********
# 模型
model = large_B_model.NN_E(target_field_id)
# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model = model.cuda()
if flag_data_refer:
    path_refer = path_save + name_refer
    model.load_state_dict(torch.load(path_refer))
# 优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 损失函数
# criterion_0 = nn.L1Loss(reduction='sum')
criterion_1 = nn.L1Loss(reduction='sum')
criterion_2 = nn.MSELoss(reduction='sum')

# **********训练模型**********
path_model = path_save + name_pth

total_train_step = 0
best_test_loss = 100000

for i in range(epochs):
    print('第{}轮训练开始'.format(i + 1))

    model.train()
    total_train_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        train_X, train_Y2 = data
        if torch.cuda.is_available():
            train_X = train_X.cuda()
            train_Y2 = train_Y2.cuda()

        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion_1(outputs, train_Y2)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_step += 1
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(i + 1, epochs, loss)
    avg_train_loss = total_train_loss / train_num

    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for step, data in enumerate(test_bar):
            test_X, test_Y2 = data
            if torch.cuda.is_available():
                test_X = test_X.cuda()
                test_Y2 = test_Y2.cuda()

            outputs = model(test_X)
            test_loss = criterion_1(outputs, test_Y2)
            total_test_loss += test_loss
    avg_test_loss = total_test_loss / test_num

    print('[epoch %d] train_loss: %.8f  test_loss: %.8f' % (i + 1, avg_train_loss, avg_test_loss))  # 50000 0.98

    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), path_model)
        print("Trained model successfully saved, path: ", path_model)

end_time = time.time()
print("total time: ", end_time - start_time)
print("end")
