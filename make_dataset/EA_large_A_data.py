import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# EA_large_A_get_data.py
# 程序D已经制作完成了大数据集
# 程序E负责根据大数据集制作小数据集，小数据集是直接用于神经网络训练的
# 程序EA的作用，就是负责制作用于训练神经网络A的小数据集
# 为简易起见，本程序只处理大型零件
# 现在分析，制作神经网络A的小数据集，需要什么：
# 神经网络A，输入应变片应变，输出载荷
# 对于应变片位置：<clip_preserve>
# 对于应变：小数据集来源于大数据集，大数据集存放位置：<data_folder><data_file_end>
# 对于载荷：<simu_list>
# 对于小数据集存放位置：<NN_A_data_folder><NN_A_data_get_file>
# 本程序的作用：
# 读取上述这些东西，输出1个csv文件
# 注意：
# 本程序只处理大型零件
# 记得每次运行前，都务必检查part_configuration.xml的设置
# 尤其是设置好<NN_A_data_get_file>，即输出文件名
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 关于程序EA输出的这1个csv的说明：
# 举例：
# 对于大型零件：
# large_A_data.csv是n×15的
# n是样本数，15列依次是：样本编号、8个应变片应变、6个载荷
# n取决于大数据集的文件夹里放了多少个样本
# **************************************** #

# **************************************** #
# 读取xml文件
# **************************************** #
path_current = os.path.dirname(os.path.realpath(__file__))
path_xml = os.path.join(path_current, 'part_configuration.xml')
# 选择处理小件还是大件？
# 本程序只处理大件
root = ET.parse(path_xml).getroot()
part = root.find('part')
xml_msg = root.find('large')

# 读取输入文件名
path_save = xml_msg.find('path_save').text.strip()
clip_preserve = xml_msg.find('region/clip_preserve').text.strip()
simu_list = xml_msg.find('FM/simu_list').text.strip()

# 读取大数据集存放的文件夹路径
path_data_folder = xml_msg.find('job/data_folder').text.strip()
data_file_end = xml_msg.find('job/data_file_end').text.strip()

# 读取取出的用于神经网络训练的数据存放的位置
path_NN_data = xml_msg.find('NN_A/NN_A_data_folder').text.strip()
file_NN_data = xml_msg.find('NN_A/NN_A_data_file').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 神经网络A的输入：应变片相关信息
path_clip_preserve = os.path.join(path_save, clip_preserve)
data_clip = pd.read_csv(path_clip_preserve, header=None, encoding="utf-8").to_numpy()
# num_each_clip记录每个应变片测点包含多少个网格单元
num_row_clip = data_clip.shape[0]
num_each_clip = np.zeros(16)
for i in range(num_row_clip):
    clip_id = int(data_clip[i][1])
    num_each_clip[clip_id - 1] += 1
# 神经网络A的输出：力和力矩相关信息
path_simu_list = os.path.join(path_save, simu_list)
data_simu = pd.read_csv(path_simu_list, header=None, encoding="utf-8").to_numpy()

# ********************
# 提取数据，大数据集文件夹的每个文件，提出为一行输出到path_ans中
path_ans = os.path.join(path_NN_data, file_NN_data)
data_ans = []
if os.path.exists(path_data_folder):
    for file_name in os.listdir(path_data_folder):
        if file_name.endswith(data_file_end):
            str_job_id = file_name.rstrip(data_file_end)
            if str_job_id.isdigit():
                job_id = int(str_job_id)
                print('processing: ', job_id)
                path_file = os.path.join(path_data_folder, file_name)
                data_raw = pd.read_csv(path_file, usecols=['elementLabel_CENTROID', 'E11', 'E22', 'E33'], header=0,
                                       encoding="utf-8").to_numpy()
                # 应变片
                row_ans_1 = np.zeros(15)
                row_ans_2 = np.zeros(15)
                row_ans_1[0] = job_id
                row_ans_2[0] = job_id
                # 应变片
                for i in range(num_row_clip):
                    clip_id = int(data_clip[i][1])
                    prsv_id = int(data_clip[i][2])
                    # print(data_raw[prsv_id])
                    if 1 <= clip_id <= 8:
                        if clip_id == 1:
                            row_ans_1[clip_id] += data_raw[prsv_id][2]
                        elif clip_id == 5:
                            row_ans_1[clip_id] += data_raw[prsv_id][1]
                        else:
                            row_ans_1[clip_id] += data_raw[prsv_id][3]
                    elif 9 <= clip_id <= 16:
                        if clip_id == 9:
                            row_ans_2[clip_id - 8] += data_raw[prsv_id][2]
                        elif clip_id == 13:
                            row_ans_2[clip_id - 8] += data_raw[prsv_id][1]
                        else:
                            row_ans_2[clip_id - 8] += data_raw[prsv_id][3]
                for i in range(8):
                    row_ans_1[i + 1] /= num_each_clip[i]
                    row_ans_2[i + 1] /= num_each_clip[i + 8]
                # 力和力矩
                data_fm = data_simu[job_id][1:7]
                # 注意，坐标系反转过，详见程序AB
                data_fm[1] = -data_fm[1]
                data_fm[2] = -data_fm[2]
                data_fm[4] = -data_fm[4]
                data_fm[5] = -data_fm[5]
                for i in range(6):
                    row_ans_1[i + 9] = data_simu[job_id][i + 1]
                    if i == 2 or i == 5:
                        row_ans_2[i + 9] = data_simu[job_id][i + 1]
                    else:
                        row_ans_2[i + 9] = -data_simu[job_id][i + 1]

                data_ans.append(row_ans_1)
                data_ans.append(row_ans_2)

df_ans = pd.DataFrame(data_ans)
df_ans.to_csv(path_ans, header=False, index=False, encoding='utf-8')

# ********************

print('\nCompleted!')
