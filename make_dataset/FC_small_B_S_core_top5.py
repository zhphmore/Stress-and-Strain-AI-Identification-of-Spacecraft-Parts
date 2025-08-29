import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# FC_small_B_S_core_top5.py
# 碎碎念：
# 程序FC干的事，和程序EB差不多
# 程序FC的作用，也是负责制作用于训练神经网络B的小数据集
# 区别在于，程序EB制作的数据集，训练出的神经网络B，预测危险区域内所有单元
# 但程序FC制作的数据集，训练出的神经网络B，只预测危险区域的核心单元
# 因此，程序FC的使用方法类似程序EB
# 程序FC输出的csv来源于程序EB输出的csv
# 分析需要哪些东西：
# 核心单元包括哪些：<path_save><core_sort>
# 从哪里取数据：<NN_B_data_folder><NN_B_data_S_file>
# 数据放到哪里：<NN_B_data_folder><NN_B_data_S_core_top5_file>
# 本程序的作用：
# 读取上述这些东西，输出1个csv文件
# 注意：
# 本程序只处理小型零件，而且只制作神经网络B应变的数据集
# 如果要完整地制作出训练神经网络B的小数据集，本程序需要运行多次
# 记得每次运行前，都务必检查part_configuration.xml的设置，并且修改第56行的内容
# 尤其是设置好<NN_B_data_S_core_top5_file>，即输出文件名
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 阐明两点：
# 一是：
# 程序FC和程序EB都只输出1个csv，输出的csv的行数相同（样本数），列数不同（单元数，EC列更少）
# 二是：
# 逻辑上，先产生危险区域内所有单元的应变数据集，再产生核心单元的应变数据集，后者来源于前者
# 因此，程序FC输出的csv来源于程序EB输出的csv
# **************************************** #

# **************************************** #
# 运行前记得改这里
# **************************************** #
# ********************
# 改这里：区域
target_field_id = 1
# ********************

# **************************************** #
# 运行前记得改这里
# **************************************** #
# ********************
# 改这里：区域
target_field_id = 1
# ********************

# **************************************** #
# 读取xml文件
# **************************************** #
path_current = os.path.dirname(os.path.realpath(__file__))
path_xml = os.path.join(path_current, 'part_configuration.xml')
# 选择处理小件还是大件？
root = ET.parse(path_xml).getroot()
part = root.find('part')
xml_msg = root.find('small')

# 读取文件保存路径
# 注意程序运行前，请修改这里
NN_B_data_folder = xml_msg.find('NN_B/NN_B_data_folder').text.strip()
NN_B_data_S_file = xml_msg.find('NN_B/NN_B_data_S_file').text.strip()
path_save = xml_msg.find('path_save').text.strip()
file_core_sort = xml_msg.find('core/core_sort').text.strip()

# 读取输入文件名
NN_B_data_S_core_top5_file = xml_msg.find('NN_B/NN_B_data_S_core_top5_file').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
path_data = os.path.join(NN_B_data_folder, NN_B_data_S_file)
data_NN = pd.read_csv(path_data, header=None, encoding='utf-8').to_numpy()
len_NN = data_NN.shape[0]
path_crit = os.path.join(path_save, file_core_sort)
data_crit = pd.read_csv(path_crit, header=None, encoding='utf-8').to_numpy()
len_crit = data_crit.shape[0]

data_ans = np.zeros((len_NN, int(len_crit + 7)))
data_ans_each = np.zeros(int(len_crit + 7))

for i in range(len_NN):
    data_ans_each[0:7] = data_NN[i][0:7]
    for k in range(len_crit):
        elem_order = int(data_crit[k][1])
        id_ans = int(k + 7)
        id_elem_order = int(elem_order + 7)
        data_ans_each[id_ans] = data_NN[i][id_elem_order]
    data_ans[i] = data_ans_each

df_ans = pd.DataFrame(data_ans)
path_S_core = os.path.join(NN_B_data_folder, NN_B_data_S_core_top5_file)
df_ans.to_csv(path_S_core, header=False, index=False, encoding='utf-8')

# ********************

print('\nCompleted!')

