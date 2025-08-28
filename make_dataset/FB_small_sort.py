import os
import xml.etree.ElementTree as ET
import csv
import numpy as np
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# FB_small_sort.py
# 碎碎念：
# 程序FB的结果，具体定义了什么是“核心单元”
# 本程序的作用：
# 输入①<core_grade>②<region_elem>，输出<core_sort>
# 输入2个csv文件，输出1个csv文件
# 注意：
# 本程序只处理小型零件
# 由于每次运行只输出1个csv，即一次只能罗列出1个危险区域的核心单元
# 因此要罗列完小型零件的4个危险区域，本程序至少运行4次
# 记得每次运行前，都务必检查part_configuration.xml的设置，并且修改第56行的内容
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 关于程序FB输出的这1个csv的说明：
# 举例：
# small_B_sort_top5_R1.csv，格式是112×3
# 说明小型零件危险区域1有112个核心单元
# 列依次是单元编号、区域顺序编号、出现次数
# 什么是区域顺序编号？参见FA的注释
# 什么是出现次数？让我们继续用故事进行解析
# **************************************** #

# **************************************** #
# 让我们继续之前的故事：
# 回顾：
# 假设本市有4所学校，每所学校1000名学生，每所学校各需选出一部分核心学生参加竞赛
# 我们进行很多很多场考试，每次考试前5名获得竞赛资格
# 程序FA列出了每场考试每所学校的前5名，相当于成绩单
# 程序FB就是根据成绩单，列出遴选出的学生名单
# 好，继续：
# 假设小张在本市所有学生中的编号为3141，在他的学校的学号为626，在所有考试中考进过前5名64次
# 那么小张的单元编号、区域顺序编号、出现次数依次为3141、626、64
# 假设1号学校共选出了112名核心学生参加竞赛
# 那么输出的csv就是112行3列的，列依次是各核心学生的单元编号、区域顺序编号、出现次数
# 这112行按出现次数从大到小排序
# 也就是某名学生考进过越多前5名，他的名字应排在越前面
# 反映到本项目：
# 一个网格单元的mises应力出现在前5名的次数越多，它就越重要，应排在越前面
# **************************************** #

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
path_save = xml_msg.find('path_save').text.strip()

# 读取输入文件名
region_elem_ordered = xml_msg.find('region/region_elem').text.strip()
file_core_grade = xml_msg.find('core/core_grade').text.strip()

file_core_sort = xml_msg.find('core/core_sort').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
path_core_grade = os.path.join(path_save, file_core_grade)

data_core_grade = pd.read_csv(path_core_grade, header=None, encoding="utf-8").to_numpy()
len_sample = int(data_core_grade.shape[0])

data_id = np.zeros(len_sample * 5)

ct = 0
col_region_start = int(5 * (target_field_id - 1) + 1)
for i in range(len_sample):
    for j in range(5):
        data_id[ct] = data_core_grade[i][j + col_region_start]
        ct += 1

df_data_id = pd.DataFrame(data_id)
data_id_count = df_data_id.value_counts()

# **********
path_region_elem_ordered = os.path.join(path_save, region_elem_ordered)
data_region_elem_ordered = pd.read_csv(path_region_elem_ordered, header=None, encoding="utf-8").to_numpy()
num_region_row = [0, 1016, 2050, 3059]

path_core_sort = os.path.join(path_save, file_core_sort)
with open(path_core_sort, mode='w', encoding='utf8', newline='') as f_count:
    writer_count = csv.writer(f_count)
    for key in data_id_count.keys():
        str_key_record = str(key).lstrip('\"(').rstrip(',)\"')

        num_key_record = int(float(str_key_record))

        elem_id = data_region_elem_ordered[int(num_region_row[int(target_field_id - 1)] + num_key_record)][1]

        line_char = [elem_id, str_key_record, str(data_id_count[key])]
        writer_count.writerow(line_char)

# ********************

print('\nCompleted!')
