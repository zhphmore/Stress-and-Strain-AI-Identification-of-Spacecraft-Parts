import sys
import os
import xml.etree.ElementTree as ET

# ******************************************************************************** #
# ******************************************************************************** #
# CC_delete_not_odb.py
# 准备工作：
# 用run_inp.py批量运行完了文件夹<odb_folder>的所有inp文件（可能需要运行很久，1万个inp文件至少一天）
# 碎碎念：
# 此时文件夹里有大量文件，但只有odb文件是需要的
# 这些不需要的文件很占硬盘空间的
# 程序CC的作用，就是删去文件夹里的其它不要的东西，只留下odb
# 本程序作用：
# 遍历文件夹<odb_folder>，除了odb文件，都删咯
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

# 读取需要被处理的文件夹
path_odb_folder = xml_msg.find('job/odb_folder').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 删除文件夹内非odb文件
# 需要保留的文件的后缀
save_ex = '.odb'
# save_ex = '.inp'

cnt = 0
# 删除不是odb的文件，并统计删除了多少个文件
for root, dirs, files in os.walk(path_odb_folder):
    for file in files:
        # 检查每一个文件的后缀
        file_check = os.path.join(root, file)
        if not file_check.endswith(save_ex):
            os.remove(file_check)
            cnt += 1

print('Delete {} files'.format(cnt))

# ********************

print('\nCompleted!')
