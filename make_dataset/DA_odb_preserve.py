import sys
import os
import xml.etree.ElementTree as ET

# ******************************************************************************** #
# ******************************************************************************** #
# DA_odb_preserve.py
# 碎碎念：
# 非常好，程序C搞完了abaqus的批量仿真，留下了一堆odb文件
# 程序D就是从odb提出数据，保存到csv，制作成大数据集
# 但是，从odb提出数据需要abaqus，可以直接用py文件调用abaqus
# odb_preserve.py就是负责调用abaqus，让其从odb提出数据的
# odb_preserve.py具体怎么发挥还要看程序DB的
# 程序DA的作用，就是搞出一个odb_preserve.py
# 很好，那么odb_preserve.py需要什么呢？
# 从odb提出数据，肯定要知道从哪儿提，然后提到哪儿呀
# 从哪儿提：odb放在哪个文件夹<odb_folder>
# 提到哪儿：提出的数据放在哪个文件夹<data_folder>，提出的数据csv文件起什么名<data_file_end>
# 另外，请回忆“有用单元”这个概念（如果忘了请看程序AC的注释）
# 一个零件有太多的网格单元了，如果全部保存下来，咱的硬盘都装不下
# 所以咱们的大数据集只需要保存“有用”的那部分网格单元，所以还需要<element_preserve>
# 本程序作用：
# 读取模板py文件odb_preserve_TMPL.py，以其为模板，生成odb_preserve.py
# 因此本程序会输出一个py文件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 举例：
# 若对于小型零件，<data_file_end>填_SEY_small.csv
# 那么对于编号511的样本Job-511.odb，提取出来的数据文件叫511_SEY_small.csv
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

# 读取输入文件名
path_save = xml_msg.find('path_save').text.strip()
element_preserve = xml_msg.find('mesh/element_preserve').text.strip()

# 读取odb文件夹路径
path_odb_folder = xml_msg.find('job/odb_folder').text.strip()

# 读取取出的数据存放的文件夹路径
path_data_folder = xml_msg.find('job/data_folder').text.strip()
data_file_name = xml_msg.find('job/data_file_end').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
path_current = os.path.dirname(os.path.realpath(__file__))
# 以odb_preserve_TMPL.py为模板，生成odb_preserve.py
py_TBD = ['TBDSAVEFOLDER', 'TBDELEMPRSV', 'TBDODBFOLDER', 'TBDDATAFOLDER', 'TBDDATAFILE']

path_tmpl_py = os.path.join(path_current, 'odb_preserve_TMPL.py')
with open(path_tmpl_py, mode='r') as f_template_py:
    template_py_content = f_template_py.read()
    content_new_py = template_py_content.replace(py_TBD[0], path_save)
    content_new_py = content_new_py.replace(py_TBD[1], element_preserve)
    content_new_py = content_new_py.replace(py_TBD[2], path_odb_folder)
    content_new_py = content_new_py.replace(py_TBD[3], path_data_folder)
    content_new_py = content_new_py.replace(py_TBD[4], data_file_name)
    path_new_py = os.path.join(path_current, 'odb_preserve.py')
    # 输出odb_preserve.py
    with open(path_new_py, "w") as f_new_py:
        f_new_py.write(content_new_py)
    print('成功生成：odb_preserve.py    保存位置： {}'.format(path_new_py))

# ********************

print('\nCompleted!')
