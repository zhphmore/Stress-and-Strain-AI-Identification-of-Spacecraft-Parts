import sys
import os
import subprocess
import xml.etree.ElementTree as ET

# ******************************************************************************** #
# ******************************************************************************** #
# DB_get_odb_data.py
# 刚刚程序DA搞出了个odb_preserve.py
# odb_preserve.py就是负责调用abaqus从odb提出数据的
# 但是，odb_preserve.py运行一次只能提取一个odb
# 所以odb_preserve.py需要重复运行很多次，而且必须并行加快速度
# 程序DB的作用，就是重复并行运行odb_preserve.py
# 具体实现方式是运行DB.bat，DB.bat自动调用DB_get_odb_data.py
# 注意：
# 你什么都不需要做，只需要在part_configuration.xml设置好，然后运行DB.bat
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 碎碎念：
# 举例：
# 如果<job_id_start>是2，<job_id_end>是8，
# 那么将提取从Job-2.odb到Job-8.odb的共7个odb文件，重复并行运行odb_preserve.py共7次
# 把max_num_subprocess设置为20，最多同时并行20个
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

# 读取odb文件夹路径
path_odb_folder = xml_msg.find('job/odb_folder').text.strip()

# 读取需要生成的inp文件的编号范围
job_id_start = int(xml_msg.find('job/job_id_start').text.strip())
job_id_end = int(xml_msg.find('job/job_id_end').text.strip())

# **************************************** #
# 执行程序
# **************************************** #
path_odb_preserve_py = os.path.join(path_current, 'odb_preserve.py')

max_num_subprocess = 20
process_pool = []

jobid = job_id_start
while jobid <= job_id_end:
    if len(process_pool) < max_num_subprocess:
        file_odb = 'Job-' + str(jobid) + '.odb'
        path_odb = os.path.join(path_odb_folder, file_odb)
        if os.path.exists(path_odb):
            str_command = 'abaqus cae noGUI=' + path_odb_preserve_py + ' -- ' + str(jobid) + '\n'
            pro = subprocess.Popen(str_command, shell=True)
            process_pool.append(pro)
            print('Job-{} submitted.'.format(str(jobid)))
        jobid += 1
    else:
        while all((p.poll() is None) for p in process_pool):
            pass
        for p in process_pool:
            if p.poll() is not None:
                process_pool.remove(p)
