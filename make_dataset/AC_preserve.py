import sys
import os
import xml.etree.ElementTree as ET
import pandas as pd

# ******************************************************************************** #
# ******************************************************************************** #
# AC_preserve.py
# 碎碎念：
# 一个零件有太多的网格单元了，如果全部保存下来，咱的硬盘都装不下
# 好在不是所有的数据都是需要的
# 零件头尾两端，既没有贴应变片，也不是危险区域，这部分的应力应变呀什么的都不用保存了
# 所以，咱们的大数据集只需要保存“有用”的那部分网格单元
# 那么，什么是“有用”或“值得保存”的网格单元呢？
# 这就是本程序干的事情了（简单理解为，掐头去尾，再切成十字形柱体，这部分就是“有用的”）
# 程序AC的工作，就是把“有用”的网格单元编号记录下来，记成一个文件
# 准备工作：
# 依次运行了AA、AB，注意AB不要重复运行
# 本程序作用：
# 根据质心坐标，确定哪些网格单元值得被大数据集保留。（例如，只需保留零件中间部分的数据，因为端部既无应变片也非危险区域）
# 输入<element_centroid>，输出<element_preserve>
# 输入1个csv文件，输出1个csv文件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 碎碎念：
# 请充分理解以下内容：
# 对于小型零件：
# 一共有127105个网格单元，但一翻掐头去尾的操作后，只有17037个网格单元是“有用”的了
# 本程序element_preserve.csv，格式是17037x1，内容是单元编号
# 对于大型零件：
# 一共有194447个网格单元，但一翻掐头去尾的操作后，只有15333个网格单元是“有用”的了
# 本程序element_preserve.csv，格式是15333x1，内容是单元编号
# 区分两个概念：
# 以大型零件为例：
# 1、单元编号，值域[1,194447]，指所有网格单元各自的编号
# 2、有用编号，值域[0,15332]，对有用的那部分网格单元（保留下来的）重新从0开始编号
# 例如，element_preserve.csv的第12行内容是30：单元编号为30的网格单元的有用编号是11
# 例如，element_preserve.csv的第9999行内容是20105：单元编号为20105的网格单元的有用编号是9998
# “有用编号”的本质是，只考虑最终在硬盘上被保存下来的网格单元（也就是被制作成了大数据集的网格单元）
# “有用编号”这一概念将在后续程序中多次应用
# **************************************** #

# **************************************** #
# 函数，判断该网格单元是否值得保留
# **************************************** #
# 小型零件（总长32mm），只保留零件中部长度13mm，且距离x轴或y轴2mm的区域
def deserve_preserve_small(data_cordinate):
    z_min = 0
    z_max = 32
    # lz_abandon = 0.0095
    lz_abandon = 9.5
    z_low = z_min + lz_abandon
    z_high = z_max - lz_abandon
    # xy_wide = 0.001
    xy_wide = 1
    # 抛弃零件两端长度9.5mm的区域，只保留零件中部长度13mm的区域
    if z_low < data_cordinate[2] < z_high:
        # 只保留距离x轴或y轴2mm的区域，即十字形区域，十字形宽度为4mm
        if (abs(data_cordinate[0]) < xy_wide) or (abs(data_cordinate[1]) < xy_wide):
            return True
    return False


# 大型零件（总长55mm），只保留零件中部长度24mm，且距离x轴或y轴3.25mm的区域
def deserve_preserve_large(data_cordinate):
    z_min = 0
    z_max = 55
    lz_abandon = 15.5
    z_low = z_min + lz_abandon
    z_high = z_max - lz_abandon
    xy_wide = 3.25
    # 抛弃零件两端长度15.5mm的区域，只保留零件中部长度24mm的区域
    if z_low < data_cordinate[2] < z_high:
        # 只保留距离x轴或y轴3.25mm的区域，即十字形区域，十字形宽度为6.5mm
        if (abs(data_cordinate[0]) < xy_wide) or (abs(data_cordinate[1]) < xy_wide):
            return True
    return False


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
    f_deserve_preserve = deserve_preserve_small
elif part.text == 'large':
    xml_msg = root.find('large')
    f_deserve_preserve = deserve_preserve_large
else:
    sys.exit('xml wrong! Please check <part>!')

# 读取文件保存路径
path_save = xml_msg.find('path_save').text.strip()

# 读取输入文件名
element_centroid = xml_msg.find('mesh/element_centroid').text.strip()

# 读取输出文件名
element_preserve = xml_msg.find('mesh/element_preserve').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 读取element_centroid.csv，将判断哪些网格单元有价值，生成element_preserve.csv
path_element_centroid = os.path.join(path_save, element_centroid)
data_element_centroid = pd.read_csv(path_element_centroid, header=None, encoding='utf-8').to_numpy()

num_elem = data_element_centroid.shape[0]

data_element_preserve = []
# 遍历各网格单元，通过网格单元的质心坐标，判断值不值得保留
for i in range(num_elem):
    if f_deserve_preserve(data_element_centroid[i][1:]):
        data_element_preserve.append(int(data_element_centroid[i][0]))
df_element_preserve = pd.DataFrame(data_element_preserve)

# 保存值得保留的网格单元编号到文件
path_element_preserve = os.path.join(path_save, element_preserve)
df_element_preserve.to_csv(path_element_preserve, header=False, index=False, encoding='utf-8')
print('成功输出：值得保留的网格单元编号    保存位置： {}'.format(path_element_preserve))

# ********************

print('\nCompleted!')

