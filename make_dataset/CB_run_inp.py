import sys
import os
import xml.etree.ElementTree as ET
import csv

# ******************************************************************************** #
# ******************************************************************************** #
# CB_run_inp.py
# 碎碎念：
# 程序CB要干两件事
# 一是根据CA的计划，批量生成inp文件
# inp文件太多了，一次全部生成完吃不消，所以可能要多次运行CB，每次只生成一部分
# 至于你说现在，马上，要生成多少组呢？可以自己设定（设置<job_id_start>和<job_id_end>）
# 二是生成了大量inp文件后，还需要使用abaqus批量并行运行inp文件
# 可以直接用py文件调用abaqus，即run_inp.py
# 具体实现方式是运行run_inp.bat，run_inp.bat自动调用run_inp.py
# 于是程序CB还需要生成run_inp.py
# 注意，程序CB运行完后，还要把生成的run_inp.py和run_inp.bat一起复制粘贴到<odb_folder>文件夹里
# 准备工作：
# 运行了CA，提前新建一个文件夹，把该文件夹名记录在<odb_folder>（该文件夹用于存放即将生成的大量inp文件）
# 本程序作用：
# 首先读取模板inp文件，以其为模板，在一个文件夹里输出大量inp文件
# 然后读取模板py文件run_inp_TMPL.py，以其为模板，生成run_inp.py
# 输入①<inp_template>，②<simu_list>，输出①大量inp文件，②run_inp.py
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 碎碎念：
# 举例：
# 对于大型零件的第一批次数据，运行后保留51959组，抛弃2472组
# 那么，simu_list_large_1.csv是51960×7的（第1行是0,0,0,0,0,0,0，读csv时不用读进去）
# 如果<job_id_start>是2，<job_id_end>是8，那么将生成样本编号为2到8的共7个inp文件，对应simu_list_large_1.csv的第3到第9行
# 举例：
# 对于大型零件的第一批次数据，总共需要生成51959个inp文件
# 笔者一般一次最多生成10000个，总共运行了6次CB，装了6个文件夹
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
inp_template = xml_msg.find('inp/inp_template').text.strip()
simu_list = xml_msg.find('FM/simu_list').text.strip()

# 本程序将在此文件夹内生成inp文件
path_inp_folder = xml_msg.find('job/odb_folder').text.strip()

# 读取需要生成的inp文件的编号范围
job_id_start = int(xml_msg.find('job/job_id_start').text.strip())
job_id_end = int(xml_msg.find('job/job_id_end').text.strip())

# **************************************** #
# 执行程序
# **************************************** #
# job_num = job_id_end - job_id_start + 1
# ********************
# 读取模板inp文件和simu_list.csv，在指定文件夹下批量生成inp文件
path_template = os.path.join(path_save, inp_template)
path_simu_list = os.path.join(path_save, simu_list)

# 模板inp文件中需要替换的字符串
char_TBD = ['TBDTBDID', 'TBDTBDFX', 'TBDTBDFY', 'TBDTBDFZ', 'TBDTBDMX', 'TBDTBDMY', 'TBDTBDMZ']

# 以“只读”方式读取模板inp文件Job_TMPL.inp
with open(path_template, mode='r') as f_template:
    template_content = f_template.read()
    # 以“只读”方式读取simu_list.csv
    with open(path_simu_list, mode='r', encoding='utf8', newline='') as f_record:
        reader_record = csv.reader(f_record)

        # 注意：ird是从0开始的
        # 注意：simu_list.csv也是从0开始的，即第一行是Job-0，但是Job-0各力和力矩都是0，不需要仿真
        # 需要仿真的数据从Job-1开始，也就是从第二行开始
        # 即Job-n在simu_list.csv的第n+1行
        for i, row_read in enumerate(reader_record):
            # 举例：
            # 想生成从Job-101到Job-200的inp文件
            # 对应读取simu_list.csv的第102到201行，即对应ird的数字是101到200
            # 即设置job_id_start为101，job_num为100
            if job_id_start <= i <= job_id_end:
                # 设置新的inp文件的Job序列号
                content_new = template_content.replace(char_TBD[0], row_read[0])
                # 设置新的inp文件的力和力矩
                for ire in range(6):
                    content_new = content_new.replace(char_TBD[ire + 1], row_read[ire + 1])
                # 另存为新的inp文件
                # 另存位置在path_folder设置的文件夹
                file_new = 'Job-' + row_read[0] + '.inp'
                path_new = os.path.join(path_inp_folder, file_new)
                # 输出inp
                with open(path_new, "w") as f_new:
                    f_new.write(content_new)
                print(file_new, ' is generated.')

print('inp generated! : {}'.format(path_inp_folder))
print('Job id from {} to {} .'.format(job_id_start, job_id_end))

# ********************
# 以CB_run_inp_TMPL.py为模板，生成run_inp.py
py_TBD = ['TBDPATHINPFOLDER', 'TBDJOBIDSTART', 'TBDJOBIDEND']

path_tmpl_py = os.path.join(path_current, 'run_inp_TMPL.py')
with open(path_tmpl_py, mode='r') as f_template_py:
    template_py_content = f_template_py.read()
    content_new_py = template_py_content.replace(py_TBD[0], path_inp_folder)
    content_new_py = content_new_py.replace(py_TBD[1], str(job_id_start))
    content_new_py = content_new_py.replace(py_TBD[2], str(job_id_end))
    path_new_py = os.path.join(path_inp_folder, 'run_inp.py')
    # 输出run_inp.py
    with open(path_new_py, "w") as f_new_py:
        f_new_py.write(content_new_py)
    print('成功生成：run_inp.py    保存位置： {}'.format(path_new_py))

# ********************

print('\nCompleted!')
