import sys
import os
import xml.etree.ElementTree as ET
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# BA_clip_preserve.py
# 准备工作：
# 自行提前准备clip.csv（clip_small.csv和clip_large.csv）
# 碎碎念：
# clip.csv是需要我们自己手写自己提前准备的，内容为两列，依次是单元编号、应变片编号
# clip.csv说明了各编号的应变片覆盖了哪些网格单元
# 也可以说，clip.csv本身定义了应变片的形状和大小
# 例如：如果编号1的应变片在clip.csv占了6行，也就是编号1的应变片覆盖了6个网格单元，也就是编号1的应变片有6个网格单元那么大
# 程序BA，就是给clip.csv再添一列有用编号，然后另存为clip_preserve.csv
# 至于什么是“有用编号”，请查阅程序AC的注释
# 本程序作用：
# 输入clip.csv，输出clip_preserve.csv
# 输入1个csv文件，输出1个csv文件
# 输入的csv文件有2列，输出的csv文件有3列
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 读取xml文件
# **************************************** #
path_current = os.path.dirname(os.path.realpath(__file__))
path_xml = os.path.join(path_current, 'part_configuration.xml')
# 选择处理小件还是大件？
root = ET.parse(path_xml).getroot()
part = root.find('part')
if part.text == 'small':
    xml_msg = root.find('small')
elif part.text == 'large':
    xml_msg = root.find('large')
else:
    sys.exit('xml wrong! Please check <part>!')

# 读取文件保存路径
path_save = xml_msg.find('path_save').text.strip()

# 读取输入文件名
clip = xml_msg.find('region/clip').text.strip()
element_preserve = xml_msg.find('mesh/element_preserve').text.strip()

# 读取输出文件名
clip_preserve = xml_msg.find('region/clip_preserve').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 读取clip.csv和element_preserve.csv，记录各应变片测点包含的网格单元编号，并判断该编号对应大数据集哪一行，生成clip_preserve.csv
path_clip = os.path.join(path_save, clip)
data_clip = pd.read_csv(path_clip, header=None, encoding='utf-8').to_numpy()
path_element_preserve = os.path.join(path_save, element_preserve)
data_element_preserve = pd.read_csv(path_element_preserve, header=None, encoding='utf-8').to_numpy()

num_row_clip = data_clip.shape[0]
num_preserve = data_element_preserve.shape[0]

# data_clip是输入，data_clip_preserve是输出
# data_clip_preserve和data_clip行数相同，但前者比后者多一列
data_clip_preserve = []
for i in range(num_row_clip):
    for j in range(num_preserve):
        if int(data_clip[i][0]) == int(data_element_preserve[j][0]):
            # 三列依次是单元编号、应变片编号、有用编号
            data_clip_preserve.append([int(data_clip[i][0]), int(data_clip[i][1]), int(j)])
df_clip_preserve = pd.DataFrame(data_clip_preserve)

# 保存各应变片测点包含的网格单元编号及对应行数到文件
path_clip_preserve = os.path.join(path_save, clip_preserve)
df_clip_preserve.to_csv(path_clip_preserve, header=False, index=False, encoding='utf-8')
print('成功输出：各应变片测点包含的网格单元编号及对应行数    保存位置： {}'.format(path_clip_preserve))

# ********************

print('\nCompleted!')
