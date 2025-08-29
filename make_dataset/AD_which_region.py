import sys
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# ******************************************************************************** #
# ******************************************************************************** #
# AD_which_region.py
# 碎碎念：
# 神经网络预测，不需要预测所有单元，只需要预测危险区域的单元，称其为“关键单元”（或“危险单元”）
# 这里从“有用”的网格单元里，抽出一批“关键”的网格单元，并称其为“关键区域”（也就是危险区域）
# 零件有多个危险区域（小型零件4个危险区域，大型零件8个危险区域）
# 怎么判断这个单元是不是危险区域的单元呢？
# 咱们提前手动划分了危险区域的位置范围，看看单元在不在位置范围里就行了
# 判断位置需要坐标，单元是有体积的，把它体心的坐标当成是单元的位置（质心坐标）
# 程序AD的工作，就是判断各网格单元属不属于危险区域，属于哪个危险区域
# 准备工作：
# 依次运行了AA、AB、AC，注意AB不要重复运行
# 本程序作用：
# 零件有多个危险区域，判断各区域包含哪些网格单元编号。（看质心坐标是否在危险区域范围内）
# 输入①<element_preserve>，②<element_centroid>，输出<region_elem>
# 输入2个csv文件，输出1个csv文件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 碎碎念：
# 请充分理解以下内容：
# 对于小型零件：
# 一共127105个网格单元，其中17037个是“有用”的，再其中4087个是隶属危险区域的
# 在这4087个中，隶属危险区域1号到4号的各有1016、1034、1009、1028个
# 本程序输出region_elem.csv，格式是4087x3，三列依次是单元编号、区域编号、有用编号
# 单元编号值域[1,127105]，区域编号值域[1,4]，有用编号值域[0,17036]
# 对于大型零件：
# 一共194447个网格单元，其中15333个是“有用”的，再其中4693个是隶属危险区域的
# 在这4693个中，隶属危险区域1号到8号的各有735、457、453、705、730、455、455、703个
# 本程序输出region_elem.csv，格式是4693x3，三列依次是单元编号、区域编号、有用编号
# 单元编号值域[1,194447]，区域编号值域[1,8]，有用编号值域[0,15332]
# 备注：
# 若对“有用编号”在这一概念不清楚，请查阅程序AC的注释
# **************************************** #

# **************************************** #
# 函数，判断该网格单元属于哪个危险区域
# **************************************** #
# 简单来讲，函数输入坐标，输出区域编号
# 函数返回值是区域编号
# 小型零件
def which_region_small(data_cordinate):
    # 小型零件，外半径rout(8mm)，零件高度h(32mm)
    rout = 8
    h = 32
    # 小型零件，包裹危险区域的立方体空间尺寸为la×lb×lc(3×4.2×3.5mm)
    la = 3
    lb = 4.2
    lc = 3.5
    # 小型零件，危险区域距底面高度
    h_danger = 13.5
    # 小型零件4个危险区域，各自的范围
    # 危险区域编号1、2、3、4依次在：y轴负半轴、x轴正半轴、y轴正半轴、x轴负半轴
    cube_border = [[-la / 2, la / 2, -rout, -rout + lb, h_danger, h_danger + lc],
                   [rout - lb, rout, -la / 2, la / 2, h - h_danger - lc, h - h_danger],
                   [-la / 2, la / 2, rout - lb, rout, h_danger, h_danger + lc],
                   [-rout, -rout + lb, -la / 2, la / 2, h - h_danger - lc, h - h_danger]]

    bool_isinrange = False
    region_id = 0
    # 依次判断是否属于4个危险区域
    for i in range(4):
        bool_region = (cube_border[i][0] < data_cordinate[0] < cube_border[i][1]) and (
                cube_border[i][2] < data_cordinate[1] < cube_border[i][3]) and (
                              cube_border[i][4] < data_cordinate[2] < cube_border[i][5])
        if bool_region:
            bool_isinrange = True
            region_id = int(i + 1)
            break
    return bool_isinrange, region_id


# 大型零件
def which_region_large(data_cordinate):
    # 大型零件，外半径15mm，内半径10mm，零件高度h(55mm)
    rout = 15
    h = 55
    # 小型零件，包裹危险区域的立方体空间尺寸为la×lb×lc(4.5×6×3mm, 4.5×6×4mm)
    lb = 6
    lc1 = 3
    lc2 = 4
    # 小型零件，危险区域距底面高度
    h_danger1 = 15.5
    h_danger2 = 20
    # 直接认为，若x坐标或y坐标的绝对值大于ld，且z坐标处在符合范围的值内，则处于危险区域
    ld = rout - lb
    # 危险区域编号12、34、56、78依次在：y轴正半轴、x轴正半轴、y轴负半轴、x轴负半轴

    bool_isinrange = False
    region_id = 0
    # 依次判断是否属于8个危险区域
    # 危险区域编号12，y轴正半轴
    if data_cordinate[1] > ld:
        if h - h_danger1 - lc1 < data_cordinate[2] < h - h_danger1:
            bool_isinrange = True
            region_id = 1
        elif h_danger2 < data_cordinate[2] < h_danger2 + lc2:
            bool_isinrange = True
            region_id = 2
    # 危险区域编号34，x轴正半轴
    elif data_cordinate[0] > ld:
        if h - h_danger2 - lc2 < data_cordinate[2] < h - h_danger2:
            bool_isinrange = True
            region_id = 3
        elif h_danger1 < data_cordinate[2] < h_danger1 + lc1:
            bool_isinrange = True
            region_id = 4
    # 危险区域编号56，y轴负半轴
    elif data_cordinate[1] < -ld:
        if h - h_danger1 - lc1 < data_cordinate[2] < h - h_danger1:
            bool_isinrange = True
            region_id = 5
        elif h_danger2 < data_cordinate[2] < h_danger2 + lc2:
            bool_isinrange = True
            region_id = 6
    # 危险区域编号78，x轴负半轴
    elif data_cordinate[0] < -ld:
        if h - h_danger2 - lc2 < data_cordinate[2] < h - h_danger2:
            bool_isinrange = True
            region_id = 7
        elif h_danger1 < data_cordinate[2] < h_danger1 + lc1:
            bool_isinrange = True
            region_id = 8
    return bool_isinrange, region_id


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
    f_which_region = which_region_small
elif part.text == 'large':
    xml_msg = root.find('large')
    f_which_region = which_region_large
else:
    sys.exit('xml wrong! Please check <part>!')

# 读取文件保存路径
path_save = xml_msg.find('path_save').text.strip()

# 读取输入文件名
element_centroid = xml_msg.find('mesh/element_centroid').text.strip()
element_preserve = xml_msg.find('mesh/element_preserve').text.strip()

# 读取输出文件名
region_elem = xml_msg.find('region/region_elem').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 读取element_centroid.csv和element_preserve.csv，计算各危险区域包含的网格单元编号，生成region_elem.csv
path_element_centroid = os.path.join(path_save, element_centroid)
data_element_centroid = pd.read_csv(path_element_centroid, header=None, encoding='utf-8').to_numpy()
path_element_preserve = os.path.join(path_save, element_preserve)
data_element_preserve = pd.read_csv(path_element_preserve, header=None, encoding='utf-8').to_numpy()
# 网格单元的数量（被保留下来的）
num_elem = data_element_preserve.shape[0]

data_region_elem = []

# 用于统计各区域包含的网格单元数量
region_cnt = np.zeros(8)
# 遍历各网格单元，通过网格单元的质心坐标，判断属于哪个区域
# 只遍历“有用”的网格单元，因此i是有用编号
# 说人话：对于大型零件，只遍历15333个，而不是194447个
for i in range(num_elem):
    # elem_id是单元编号
    elem_id = int(data_element_preserve[i][0])
    j = int(elem_id - 1)
    # 函数返回，是否属于一个危险区域，属于危险区域的编号
    bool_isinrange, region_id = f_which_region(data_element_centroid[j][1:])
    if bool_isinrange:
        region_cnt[int(region_id - 1)] += 1
        # 三列依次是单元编号、区域编号、有用编号
        # 其实int(data_element_preserve[i][0])、elem_id、int(data_element_centroid[j][0])都是单元编号，写哪个都行
        data_region_elem.append([int(data_element_centroid[j][0]), region_id, i])

# ********************
# 理论上data_region_elem就已经可以保存成region_elem.csv了
# 但为了更完美一点，还可以进行按区域编号排序
# 例如：
# data_region_elem的内容可能是：
# 13798,3,5483
# 13799,4,5484
# 13822,3,5507
# data_region_elem_ordered就会调整为：
# 13798,3,5483
# 13822,3,5507
# 13799,4,5484
# 最终得到的region_elem.csv就会更规范

data_region_elem_ordered = []

num_region_elem = len(data_region_elem)
for j in range(8):
    for i in range(num_region_elem):
        if int(data_region_elem[i][1]) == int(j + 1):
            data_region_elem_ordered.append(data_region_elem[i])

df_region_elem = pd.DataFrame(data_region_elem_ordered)

# ********************
print('各危险区域包含的网格单元数量（小件4个危险区域，大件8个危险区域）： ', region_cnt)

# 保存各危险区域包含的网格单元编号到文件
path_region_elem = os.path.join(path_save, region_elem)
df_region_elem.to_csv(path_region_elem, header=False, index=False, encoding='utf-8')
print('成功输出：各危险区域包含的网格单元编号    保存位置： {}'.format(path_region_elem))

# ********************

print('\nCompleted!')

