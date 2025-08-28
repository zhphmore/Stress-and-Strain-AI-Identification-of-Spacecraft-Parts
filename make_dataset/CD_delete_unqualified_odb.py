import sys
import os
import xml.etree.ElementTree as ET

# ******************************************************************************** #
# ******************************************************************************** #
# CD_delete_unqualified_odb.py
# 准备工作：
# 运行了CC
# 碎碎念：
# CC已经删去文件夹里除odb以外的东西，现在只剩下odb了
# 但不是所有的odb都是正常的，有时abaqus的仿真突然崩溃，这个样本的仿真就出错了导致其odb也是错的
# 不合格的odb的文件大小有明显异常，可以通过odb的文件大小来判断odb合不合格
# 程序CD的作用，就是删去不合格的odb
# 如果有不合格的odb文件，会将其记录在<bug_odb>的txt文件，然后删去
# 因此本程序会输出一个txt文件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 特别注意：
# 大小不合格的odb文件被认为是计算错误的odb文件，理应删除
# 建议：
# 第一次运行该程序时，先注释掉本程序第89行 # os.remove(file_check)
# 以避免<odb_KB>大小设置不合理而误删odb
# 检查无误后，再取消掉注释，第二次运行该程序，正式删除大小不合格的odb文件
# **************************************** #

# **************************************** #
# 碎碎念：
# 对于小型零件：
# 合格的odb大小范围在121000KB到121020KB
# 对于大型零件：
# 合格的odb大小范围在186600KB到186620KB
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

# 读取需要被处理的文件夹
path_odb_folder = xml_msg.find('job/odb_folder').text.strip()

# 读取文件保存路径
path_save = xml_msg.find('path_save').text.strip()

# 读取输出文件名
bug_odb = xml_msg.find('job/bug_odb').text.strip()

# 读取合格的odb的文件大小
# odb文件至少这么大，才被认为合格，否则认为没有odb文件在生成时，abaqus没有跑完
target_odb_KB_down = float(xml_msg.find('job/odb_KB_down').text.strip())
target_odb_KB_up = float(xml_msg.find('job/odb_KB_up').text.strip())

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 遍历路径为path_odb_folder的文件夹里的odb，生成bug_odb.txt
# 判断odb文件大小是否正常，记录大小不正常的odb的文件名在bug_odb.txt

path_bug_odb = os.path.join(path_save, bug_odb)
# 注意：是以“继续写入”的方式打开！
f_bug_odb = open(path_bug_odb, 'a')

for root, dirs, files in os.walk(path_odb_folder):
    for file in files:
        # 检查每一个文件的大小
        file_check = os.path.join(root, file)
        fsize_B = os.stat(file_check).st_size
        fsize_KB = fsize_B / 1024
        # 如果达不到这么大
        if not (target_odb_KB_down <= fsize_KB <= target_odb_KB_up):
            print(file_check, fsize_KB)
            # 记录不合格的odb文件的路径
            txt_line = file_check + '\n'
            f_bug_odb.write(txt_line)
            # 删除不合格的odb文件
            # os.remove(file_check)

f_bug_odb.close()

# ********************

print('\nCompleted!')
