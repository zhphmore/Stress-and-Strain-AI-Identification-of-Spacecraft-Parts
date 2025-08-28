import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# EB_small_B_E.py
# 程序EB的作用，就是负责制作用于训练神经网络B的小数据集
# 为简易起见，本程序只处理小型零件
# 本程序只制作应变的数据集（应力是另一个程序EB_small_B_S.py）
# 现在分析，制作神经网络B的小数据集，需要什么：
# 神经网络B，输入载荷，输出应力或应变
# 对于载荷：<simu_list>
# 对于应力或应变：小数据集来源于大数据集，大数据集存放位置：<data_folder><data_file_end>
# 对于危险区域信息：<region_elem>
# 对于小数据集存放位置：<NN_B_data_folder><NN_B_data_E_file>
# 本程序的作用：
# 读取上述这些东西，输出1个csv文件
# 注意：
# 本程序只处理小型零件，而且只制作神经网络B应变的数据集
# 如果要完整地制作出训练神经网络B的小数据集，本程序需要运行多次
# 记得每次运行前，都务必检查part_configuration.xml的设置，并且修改第77行的内容
# 尤其是设置好<NN_B_data_E_file>，即输出文件名
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 碎碎念：
# 训练神经网络B的小数据集，细节之多，容我娓娓道来
# ① 什么是危险区域？
# 如果忘记了危险区域的概念，建议查阅程序AD的注释
# 神经网络B预测，不需要预测所有单元（否则预测量太大），只需要预测危险区域的单元（小型零件有4个危险区域）
# 因此神经网络B分成了4个子网络，每个子网络只负责预测其中一个危险区域
# 这4个危险区域各自有多大呢？这由<region_elem>负责，该文件之前已由程序AD产生
# ② 应力与应变
# 为提高预测精细度，要把应力与应变分开预测
# 因此实际需要训练8个子网络（前4个预测4个危险区域的应力，后4个预测4个危险区域的应变）
# 8个子网络需要8个数据集，EB_small_B_S.py制作4个，EB_small_B_E.py制作4个
# 因此本程序至少运行4次
# ③ 本程序要修改什么？
# 若要产生不同区域的数据集，请修改第77行的内容
# ④ 本程序实际需要运行多少次？
# 刚刚说了，神经网络B的应变预测部分包括4个子网络，需要4个数据集，因此本程序至少运行4次
# 但是，可能制作1个数据集，就要不止一次运行本程序
# 举例：
# 假如大数据集分开装了3个文件夹
# 由于小数据集是由大数据集产生的，<data_folder>只能写一个文件夹
# 所以要产生一个小数据集就要运行本程序3次，每次修改<data_folder>的内容
# 听不懂吗？我这样讲吧：
# 第1次运行，产生small_B_E_R1_1.csv是n1×6103的
# 第2次运行，产生small_B_E_R1_2.csv是n2×6103的
# 第3次运行，产生small_B_E_R1_3.csv是n3×6103的
# 三次结果拼接，产生small_B_E_R1.csv是62234×6103的（n1+n2+n3=62234）
# 所以要产生一个小数据集就要运行本程序3次（3次是举例，实际多少次看情况，看大数据集样本装了几个文件夹）
# 那么对于4个子网络，就要运行本程序共12次
# 所以你知道制作数据集有多辛苦了吧！
# **************************************** #

# **************************************** #
# 关于程序EB输出的这1个csv的说明：
# 举例：
# 对于小型零件，危险区域1的应变的数据集：
# 理论上：
# small_B_E_R1.csv是62234×6103的
# 62234是样本数，6103列依次是：样本编号、6个载荷、6个应变分量×1016个网格单元（危险区域1的）
# 实际上：
# small_B_E_R1.csv是n×6103的
# n取决于大数据集的文件夹里放了多少个样本
# **************************************** #

# **************************************** #
# 运行前记得改这里
# **************************************** #
# ********************
# 改这里：区域
target_field_id = 1
# ********************
# 神经网络B的输出：各区域有多少网格单元
num_each_region = [1016, 1034, 1009, 1028]
num_this_region = num_each_region[int(target_field_id - 1)]

# **************************************** #
# 读取xml文件
# **************************************** #
path_current = os.path.dirname(os.path.realpath(__file__))
path_xml = os.path.join(path_current, 'part_configuration.xml')
# 选择处理小件还是大件？
root = ET.parse(path_xml).getroot()
part = root.find('part')
xml_msg = root.find('small')

# 读取输入文件名
path_save = xml_msg.find('path_save').text.strip()
simu_list = xml_msg.find('FM/simu_list').text.strip()
region_elem = xml_msg.find('region/region_elem').text.strip()

# 读取大数据集存放的文件夹路径
path_data_folder = xml_msg.find('job/data_folder').text.strip()
data_file_end = xml_msg.find('job/data_file_end').text.strip()

# 读取取出的用于神经网络训练的数据存放的位置
path_NN_data = xml_msg.find('NN_B/NN_B_data_folder').text.strip()
file_NN_data = xml_msg.find('NN_B/NN_B_data_E_file').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 神经网络B的输入：力和力矩相关信息
path_simu_list = os.path.join(path_save, simu_list)
data_simu = pd.read_csv(path_simu_list, header=None, encoding="utf-8").to_numpy()

# 神经网络B的输出：各危险区域包含的网格单元数量
path_region_elem = os.path.join(path_save, region_elem)
data_region = pd.read_csv(path_region_elem, header=None, encoding="utf-8").to_numpy()
# 所有网格单元一共有多少个网格单元
num_row_region = data_region.shape[0]

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
                data_raw = pd.read_csv(path_file,
                                       usecols=['E11', 'E22', 'E33', 'E12', 'E13', 'E23'],
                                       header=0,
                                       encoding="utf-8").to_numpy()

                row_ans_1 = np.zeros(int(6 * num_this_region + 7))
                row_ans_1[0] = job_id

                # 力和力矩
                for i in range(6):
                    row_ans_1[i + 1] = data_simu[job_id][i + 1]

                # 应变
                ct = 0
                for i in range(num_row_region):
                    field_id = int(data_region[i][1])
                    if field_id == int(target_field_id):
                        prsv_id = int(data_region[i][2])
                        row_ans_1[int(ct * 6 + 7):int(ct * 6 + 13)] = data_raw[prsv_id][0:6]
                        ct += 1

                data_ans.append(row_ans_1)

df_ans = pd.DataFrame(data_ans)
df_ans.to_csv(path_ans, header=False, index=False, encoding='utf-8')

# ********************

print('\nCompleted!')
