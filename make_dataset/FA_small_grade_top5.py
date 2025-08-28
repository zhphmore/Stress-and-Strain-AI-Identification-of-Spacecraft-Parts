import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# FA_small_grade_top5.py
# 故事看起来在程序E就结束了，但需求总是在变的，这不，新计划就来了
# 神经网络B只需要预测危险区域的单元，但这依然太多了（例如：小型零件危险区域1就有1016个），能不能再少一些？
# 我们从“危险单元”中再定义一部分单元，称之为“核心单元”，让神经网络B只预测核心单元（实际表明，小型零件危险区域1核心单元只有112个）
# 程序F的作用，就是定义哪些单元是“核心单元”
# 我们分两步完成：程序FA负责”遍历“，程序FB负责”统计“，二者结合，就回答了什么是“核心单元”
# 我知道你可能一头雾水，但坚持看完后面的故事你就懂了
# 假设你看完了故事，就用考试来类比
# 考生信息：<region_elem>
# 考试分数：mises应力来源于大数据集，大数据集存放位置：<data_folder><data_file_end>
# 考试名次存放位置：<NN_B_data_folder><core_grade>
# 本程序的作用：
# 读取上述这些东西，输出1个csv文件
# 注意：
# 本程序只处理小型零件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 先讲一个故事：
# 学校上有1000名学生，你想选出最优秀的一批学生去参加竞赛，怎么选？
# 你可以进行很多很多场考试，每次考试前5名获得竞赛资格，最终可能有100名学生考进过前5名，那么就选出了100名学生
# 我们认为，这100名学生都有争夺第1的潜力；至于其它学生，没有考第1的潜力
# 另外，一共有4所学校，每所学校都依此种方式挑选学生
# 反映到本项目：
# 1所学校就对应1个危险区域，4所学校就是小型零件的4个危险区域
# 这1000名学生就是某个危险区域的全部单元“危险单元”，遴选出的100名学生就是“核心单元”
# 我们认为，mises应力最大的单元，只可能出现在“核心单元”中，因此神经网络B只需预测“核心单元”
# 遴选出100名学生的过程，就是遴选出“核心单元”
# 对所有载荷情况进行遍历，对于每种载荷，把区域1内mises应力最大的5个单元揪出来
# 小型零件62234种载荷组合，就是进行了62234场考试
# 各学生的考试成绩，就是各网格单元的mises应力
# 程序FA干的事，就是列出这62234场考试4所学校的前5名
# **************************************** #

# **************************************** #
# 关于程序FA输出的这1个csv的说明：
# 举例：
# 理论上：
# small_B_crit_top5.csv是62234×21的
# 62234是样本数，21列依次是：样本编号、危险区域1的前5名、……、危险区域4的前5名
# 实际上：
# small_B_crit_top5.csv是n×21的
# n取决于大数据集的文件夹里放了多少个样本
# 更准确地说：
# 21列依次是：样本编号、20个区域顺序编号（4个区域各前5名）
# 什么是“区域顺序编号”？这是个新概念，下方会继续说明
# **************************************** #

# **************************************** #
# 几个概念区分：
# 单元编号、有用编号、区域顺序编号
# 以小型零件为例：
# 小型零件一共127105个网格单元，其中只有17037个网格单元被保留为大数据集，再其中危险区域1有1016个网格单元
# 单元编号值域[1,127105]，有用编号值域[0,17036]，区域顺序编号值域[0,1015]
# 在region_element_small.csv中，编号61102的网格单元的有用编号是14571，处在危险区域1的第3行位置，因此其区域顺序编号为2
# 所以该网格单元，单元编号61102、有用编号14571、区域顺序编号2
# **************************************** #

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

region_elem = xml_msg.find('region/region_elem').text.strip()

# 读取大数据集存放的文件夹路径
path_data_folder = xml_msg.find('job/data_folder').text.strip()
data_file_end = xml_msg.find('job/data_file_end').text.strip()

# 读取取出的用于神经网络训练的数据存放的位置
path_NN_data = xml_msg.find('NN_B/NN_B_data_folder').text.strip()
file_NN_data = xml_msg.find('core/core_grade').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
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
                                       usecols=['elementLabel_CENTROID', 'S_mises'],
                                       header=0,
                                       encoding="utf-8").to_numpy()

                row_ans_1 = np.zeros(21)
                row_ans_1[0] = job_id

                # 8个区域
                for k in range(4):
                    ct = 0
                    s_mises_sort = np.zeros(5)
                    s_mises_region_order = np.zeros(5)
                    # 遍历所有点，查找本区域的点
                    for i in range(num_row_region):
                        field_id = int(data_region[i][1])
                        # 如果是本区域的
                        if field_id == int(k + 1):
                            prsv_id = int(data_region[i][2])
                            # 找到前5个mises应力中最小的
                            min_s_mises_id = 0
                            min_s_mises = data_raw[int(s_mises_sort[0])][1]
                            for j in range(4):
                                if data_raw[int(s_mises_sort[j + 1])][1] < min_s_mises:
                                    min_s_mises_id = j + 1
                                    min_s_mises = data_raw[int(s_mises_sort[j + 1])][1]
                            # 准备替换
                            if data_raw[prsv_id][1] > min_s_mises:
                                s_mises_sort[min_s_mises_id] = prsv_id
                                s_mises_region_order[min_s_mises_id] = ct
                            ct += 1
                        # 如果不是本区域的
                        else:
                            continue
                    # 本区域的点查找完毕
                    # 本区域保存
                    row_ans_1[int(k * 5 + 1):int(k * 5 + 6)] = s_mises_region_order

                data_ans.append(row_ans_1)

df_ans = pd.DataFrame(data_ans)
df_ans.to_csv(path_ans, header=False, index=False, encoding='utf-8')

# ********************

print('\nCompleted!')
