import sys
import os
import xml.etree.ElementTree as ET
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# AB_coordinate.py
# 碎碎念：
# 甲方提供了零件stp文件，但是甲方建模实在太随意啦！
# 甲方建模时，小型和大型零件，单位不同（小型零件m，大型零件mm），坐标原点也有偏差
# 为了把事情做好，无奈只好收拾清楚坐标啦
# 程序AB的工作，就是重新调整坐标，包括调整原点和单位（统一为mm）
# 准备工作：
# 运行了AA，已经有了从inp里提出来的原始的<node_coordinate>和<element_centroid>
# 本程序作用：
# 输入①<node_coordinate>，②<element_centroid>
# 输出①<node_coordinate>，②<element_centroid>
# 输入2个csv文件，输出2个csv文件
# 特别注意：！！！！
# 只需在运行AA后，运行一次本程序AB，不要重复运行本程序！！！！
# 但若重新运行了AA，就要重新运行一次AB
# 也就是是每运行一次AA，就要运行一次且仅一次AB
# 重复运行的后果：
# 例如：甲方建模小型零件时，单位是m，某坐标0.002，运行AB后为2，若重复运行则变成2000了！
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

# 读取输入输出文件名
node_coordinate = xml_msg.find('mesh/node_coordinate').text.strip()
element_centroid = xml_msg.find('mesh/element_centroid').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# 读取坐标文件
# 目前所有牵涉到坐标的文件都要修正！
# 包括①各节点的坐标<node_coordinate>，②各网格单元的质心坐标<element_centroid>

# 读取node_coordinate.csv
path_node_coordinate = os.path.join(path_save, node_coordinate)
data_node_coordinate = pd.read_csv(path_node_coordinate, header=None, encoding='utf-8').to_numpy()
num_node = data_node_coordinate.shape[0]
# 读取element_centroid.csv
path_element_centroid = os.path.join(path_save, element_centroid)
data_element_centroid = pd.read_csv(path_element_centroid, header=None, encoding='utf-8').to_numpy()
num_elem = data_element_centroid.shape[0]

# 怎么调整坐标呢？
# 期望的结果：
# 1、单位应该都是mm
# 2、xy平面是横截面，z轴为旋转对称轴
# 3、z轴的原点在底面上，z轴正方向由底面指向顶面（仿真时零件底面固定，顶面施加载荷）

# 小型零件
# 原单位：m，新单位：mm
# z坐标，原范围：274-306mm，新范围：0-32mm
if part.text == 'small':
    for i in range(num_node):
        data_node_coordinate[i][1] *= 1000
        data_node_coordinate[i][2] *= 1000
        data_node_coordinate[i][3] *= 1000
        data_node_coordinate[i][3] -= 274
    for i in range(num_elem):
        data_element_centroid[i][1] *= 1000
        data_element_centroid[i][2] *= 1000
        data_element_centroid[i][3] *= 1000
        data_element_centroid[i][3] -= 274

# 大型零件
# 单位：mm，保持不变
# 零件原有问题：
# z轴的原点在顶面上，z轴正方向由顶面指向底面
# 因此，z轴正方向需要翻转，但根据右手螺旋法则：一旦z轴翻转，y轴也应该同时翻转！
# 同时z轴的原点需要从顶面调整到底面上（由于零件高55mm，因此全体z坐标移动55mm）
elif part.text == 'large':
    for i in range(num_node):
        # 翻转y轴
        data_node_coordinate[i][2] = -data_node_coordinate[i][2]
        # 翻转z轴，且z坐标移动55mm
        data_node_coordinate[i][3] = -data_node_coordinate[i][3] + 55
    for i in range(num_elem):
        data_element_centroid[i][2] = -data_element_centroid[i][2]
        data_element_centroid[i][3] = -data_element_centroid[i][3] + 55

# 坐标修正完毕
# 保存到文件，并覆盖原文件
df_node_coordinate = pd.DataFrame(data_node_coordinate)
df_node_coordinate.to_csv(path_node_coordinate, header=False, index=False, encoding='utf-8')
print('成功输出：更新后的各节点的坐标    保存位置： {}'.format(path_node_coordinate))
df_element_centroid = pd.DataFrame(data_element_centroid)
df_element_centroid.to_csv(path_element_centroid, header=False, index=False, encoding='utf-8')
print('成功输出：更新后的各网格单元的质心坐标    保存位置： {}'.format(path_element_centroid))

# ********************

print('\nCompleted!')
