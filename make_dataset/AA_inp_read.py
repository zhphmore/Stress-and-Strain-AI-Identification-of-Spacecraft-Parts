import sys
import os
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np

# ******************************************************************************** #
# ******************************************************************************** #
# AA_inp_read.py
# 碎碎念：
# 做所有事之前，总得有网格吧！用abaqus提前准备一份inp文件吧
# 如果不知道abaqus和inp的关系，建议上网搜索
# 网格信息，都直接储存在inp文件里，只需要把inp文件的某些行拷出来就可以了
# inp文件可以用记事本打开，在记事本里看看具体要拷哪些行
# 程序AA的工作，就是机械性地把这些行拷出来，保存成csv而已啦
# 准备工作：
# 准备好模板inp文件（用abaqus设置好后导出）
# 在part_configuration.xml进行设置
# 把inp文件的名称记录在：<inp_template>
# 把要拷的行的范围记录在：<node_row_start> <node_row_end> <elem_row_start> <elem_row_end>
# 本程序的作用：
# 读取模板inp文件，从中提取①各节点的坐标，②各网格单元包含的节点编号，③各网格单元的质心坐标
# 输入<inp_template>，输出①<node_coordinate>，②<element_node>，③<element_centroid>
# 输入1个inp文件，输出3个csv文件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 对于小型零件：
# 共有127105个网格单元，188191个节点
# node_coordinate.csv，格式是188191x4，列依次是节点编号、x坐标、y坐标、z坐标，对应inp文件的第10到第188200行
# element_node.csv，格式是127105x11，列依次是单元编号、单元包含的10个节点编号（C3D10），对应inp文件的第188202到第315306行
# element_centroid，格式是127105x4，列依次是单元编号、x坐标、y坐标、z坐标
# 对于大型零件：
# 共有194447个网格单元，290225个节点
# node_coordinate.csv，格式是290225x4，列依次是节点编号、x坐标、y坐标、z坐标，对应inp文件的第10到第290234行
# element_node.csv，格式是194447x11，列依次是单元编号、单元包含的10个节点编号（C3D10），对应inp文件的第290236到第484682行
# element_centroid，格式是290225x4，列依次是单元编号、x坐标、y坐标、z坐标
# **************************************** #

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

# 读取输入模板inp文件相关信息
inp_template = xml_msg.find('inp/inp_template').text.strip()
node_row_start = int(xml_msg.find('inp/node_row_start').text.strip())
node_row_end = int(xml_msg.find('inp/node_row_end').text.strip())
elem_row_start = int(xml_msg.find('inp/elem_row_start').text.strip())
elem_row_end = int(xml_msg.find('inp/elem_row_end').text.strip())

# 读取输出文件名
node_coordinate = xml_msg.find('mesh/node_coordinate').text.strip()
element_node = xml_msg.find('mesh/element_node').text.strip()
element_centroid = xml_msg.find('mesh/element_centroid').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
path_template = os.path.join(path_save, inp_template)

# ********************
# 以小型零件为例，进一步说明：
# 小型零件一共188191个结点，节点编号从inp文件的第10行开始，第188200行结束
# 对于小型零件，这部分代码的作用：
# 把inp文件第10行到第188200行拷出，另存为node_coordinate.csv
# 第10行内容：      1, 0.00282842712, -0.00282842712,  0.277499974
# 第188200行内容： 188191, 0.00427290518, 0.00615122449,  0.289653182

# 从模板inp文件提取各节点的坐标，生成node_coordinate.csv
with open(path_template, mode='r') as f_template:
    num_node = node_row_end - node_row_start + 1
    path_node_coordinate = os.path.join(path_save, node_coordinate)
    with open(path_node_coordinate, mode='w', encoding='utf8', newline='') as f_node_coordinate:
        writer_node_coordinate = csv.writer(f_node_coordinate)
        # 注意：readlines()从0开始，区间左闭右开
        # 举例：读取第1到3行：readlines()[0:3]
        lines_str = f_template.readlines()[node_row_start - 1:node_row_end]
        print('the number of nodes: ', num_node)
        print('extracting ... ')
        for i in range(num_node):
            line_str = lines_str[i].strip('\n')
            line_str = line_str.split(',')
            writer_node_coordinate.writerow(line_str)
    print('成功输出：各节点的坐标    保存位置： {}'.format(path_node_coordinate))

# ********************
# 以小型零件为例，进一步说明：
# 小型零件一共127105个网格单元，网格单元编号从inp文件的第188202行开始，第315306行结束
# 对于小型零件，这部分代码的作用：
# 把inp文件第188202行到第315306行拷出，另存为element_node.csv
# 第10行内容：  1, 11897, 11898, 11899, 11900, 25902, 25901, 25900, 25904, 25903, 25905
# 第188200行内容：127105,  25376,  24950,  25378,  24685, 184532, 176850, 188165, 176843, 174450, 176849

# 从模板inp文件提取各网格单元包含的节点编号，生成element_node.csv
with open(path_template, mode='r') as f_template:
    num_elem = elem_row_end - elem_row_start + 1
    path_element_node = os.path.join(path_save, element_node)
    with open(path_element_node, mode='w', encoding='utf8', newline='') as f_element_node:
        writer_element_node = csv.writer(f_element_node)
        lines_str = f_template.readlines()[elem_row_start - 1:elem_row_end]
        print('the number of elements: ', num_elem)
        print('extracting ... ')
        for i in range(num_elem):
            line_str = lines_str[i].strip('\n')
            line_str = line_str.split(',')
            writer_element_node.writerow(line_str)
    print('成功输出：各网格单元包含的节点编号    保存位置： {}'.format(path_element_node))

# ********************
# 这部分代码的作用：
# 利用刚刚的node_coordinate.csv和element_node.csv，搞出element_centroid.csv

# 读取刚刚生成的node_coordinate.csv和element_node.csv，计算各网格单元的质心坐标，生成element_centroid.csv
data_node_coordinate = pd.read_csv(path_node_coordinate, header=None, encoding='utf-8').to_numpy()
data_element_node = pd.read_csv(path_element_node, header=None, encoding='utf-8').to_numpy()

# 坐标多少维，三维
num_coordinate_dimension = data_node_coordinate.shape[1] - 1
# 每个网格单元多少个节点，C3D10，10个
num_elem_node = data_element_node.shape[1] - 1

# 用于存放输出数据：各网格单元的质心坐标
data_elem_centroid = np.zeros((num_elem, num_coordinate_dimension + 1))
# 遍历各网格单元
for i in range(num_elem):
    # 该网格单元的编号
    data_elem_centroid[i][0] = data_element_node[i][0]
    # 遍历该网格单元包含的节点
    for j in range(num_elem_node):
        # 该节点的坐标
        node_id = int(data_element_node[i][j + 1])
        # 把坐标在各维度上加和，三维
        for k in range(num_coordinate_dimension):
            data_elem_centroid[i][k + 1] += data_node_coordinate[node_id - 1][k + 1]
    # 该网格单元包含的节点，把坐标在各维度上平均，就是该网格单元的质心坐标
    for k in range(num_coordinate_dimension):
        data_elem_centroid[i][k + 1] /= num_elem_node
df_elem_centroid = pd.DataFrame(data_elem_centroid)

# 保存各网格单元的质心坐标到文件
path_element_centroid = os.path.join(path_save, element_centroid)
df_elem_centroid.to_csv(path_element_centroid, header=False, index=False, encoding='utf-8')
print('成功输出：各网格单元的质心坐标    保存位置： {}'.format(path_element_centroid))

# ********************

print('\nCompleted!')
