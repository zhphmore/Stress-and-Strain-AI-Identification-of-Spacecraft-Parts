import sys
import os
import csv
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
element_centroid = xml_msg.find('mesh/element_centroid').text.strip()

# 读取要绘制的文件
element_preserve = xml_msg.find('mesh/element_preserve').text.strip()
# element_preserve = xml_msg.find('region/region_elem').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
path_element_centroid = os.path.join(path_save, element_centroid)
data_element_centroid = pd.read_csv(path_element_centroid, header=None, encoding='utf-8').to_numpy()

path_element_preserve = os.path.join(path_save, element_preserve)
data_elem = pd.read_csv(path_element_preserve, header=None, encoding='utf-8').to_numpy()

# 一共多少个网格单元
num_elem = data_elem.shape[0]

cen_cor = []
# 遍历各网格单元
for i in range(num_elem):
    row_id = int(data_elem[i][0] - 1)
    cen_cor.append(
        (data_element_centroid[row_id][1], data_element_centroid[row_id][2], data_element_centroid[row_id][3]))

print(len(cen_cor))

# 绘制三维点云图
# 创建一个三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制点云
ax.scatter([p[0] for p in cen_cor], [p[1] for p in cen_cor], [p[2] for p in cen_cor], c='r', s=10)
# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# 显示图形
plt.show()
