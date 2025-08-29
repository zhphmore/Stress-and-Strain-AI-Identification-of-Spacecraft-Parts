import sys
import os
import xml.etree.ElementTree as ET
import csv

# ******************************************************************************** #
# ******************************************************************************** #
# CA_simu_list.py
# 碎碎念：
# 如果说，程序A是处理网格单元、节点、坐标的，程序B是处理应变片的
# 那么，程序C才算是正式进入abaqus的批量仿真
# abaqus批量仿真的思路：
# 如果要生成10000个数据样本，就需要10000个inp文件，多个inp文件可以并行计算
# 每一种载荷组合就对应一个样本，就对应一个inp文件
# 程序CA想好准备生成多少个inp文件，然后让程序CB一个个地生成
# 程序CA的作用，就是制定计划，准备生成多少组inp文件
# 准备工作：
# 准备好模板inp文件（用abaqus设置好后导出）
# 本程序作用：
# 读取模板inp文件，生成载荷的组合名单
# 输入<inp_template>，输出①<simu_list>，②<simu_abandon>
# 输入1个inp文件，输出2个csv文件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 碎碎念：
# 由于甲方所需的零件应力应变辨识范围，是在不发生屈服或自接触情况下的辨识
# 因此需要抛弃可能发生屈服或自接触的力和力矩组合
# 通过一个经验函数判断保留或抛弃
# 举例：
# 对于大型零件的第一批次数据，运行后保留51959组，抛弃2472组
# 那么，simu_list_large_1.csv是51960×7的（第1行是0,0,0,0,0,0,0），simu_abandon_large_1.csv是2472×7的
# 7列分别是样本编号、Fx、Fy、Fz、Mx、My、Mz
# **************************************** #

# **************************************** #
# 函数，通过经验，粗略判断某种力和力矩值的组合是否可能屈服
# **************************************** #
# 小型零件，判别式
def discriminant_small(_num_fx, _num_fy, _num_fz, _num_mx, _num_my, _num_mz) -> bool:
    if abs(_num_fx) + abs(_num_fy) + 50 * abs(_num_mx) + 50 * abs(_num_my) + 20 * abs(_num_mz) < 150:
        return True
    return False


# 大型零件，判别式
def discriminant_large(_num_fx, _num_fy, _num_fz, _num_mx, _num_my, _num_mz) -> bool:
    if 1.1 * abs(_num_fx) + 1.1 * abs(_num_fy) + 0.05 * abs(_num_mx) + 0.05 * abs(_num_my) + 0.02 * abs(_num_mz) < 600:
        return True
    return False


# **************************************** #
# 碎碎念：
# 根据甲方实际需求的调整，仿真产生数据集分了3个批次
# 对于小型零件：
# 理论上载荷共保留62479种，抛弃10382种。实际上获得62234种组合的样本。若通过数据集增强，将产生124468个样本。
# 对于大型零件
# 理论上载荷共保留68265种，抛弃2760种。实际上获得67285种组合的样本。若通过数据集增强，将产生134570个样本。
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
    f_discriminant = discriminant_small
    # 第一批次数据，运行后保留：46301组，抛弃：8130组
    # CHAR_Fx = ['0.', '-5.', '10.', '-25.', '45.', '-60.']
    # CHAR_Fy = ['0.', '5.', '-10.', '25.', '-45.', '60.']
    # CHAR_Fz = ['0.', '10.', '-50.', '100.', '-150.', '200.']
    # CHAR_Mx = ['0.', '-0.1', '0.2', '-0.4', '0.7', '-1']
    # CHAR_My = ['0.', '0.1', '-0.2', '0.4', '-0.7', '1']
    # CHAR_Mz = ['0.', '-0.1', '0.3', '-0.6', '1.', '-1.4', '2.']
    # 第二批次数据，运行后保留：12415组，抛弃：1920组
    # CHAR_Fx = ['0.', '15.', '-30.', '50.']
    # CHAR_Fy = ['0.', '-15.', '35.', '-55.']
    # CHAR_Fz = ['0.', '-20.', '40.', '-60.', '80.', '-120.', '130.', '-170.']
    # CHAR_Mx = ['0.', '0.3', '-0.5', '0.8']
    # CHAR_My = ['0.', '-0.3', '0.6', '-0.8']
    # CHAR_Mz = ['0.', '0.2', '-0.4', '0.8', '-1.', '1.5', '-1.7']
    # 第三批次数据，运行后保留：3763组，抛弃：332组
    CHAR_Fx = ['0.', '-22.', '27.', '-42.']
    CHAR_Fy = ['0.', '17.', '-32.', '37.']
    CHAR_Fz = ['0.', '70.', '-90.', '160.']
    CHAR_Mx = ['0.', '-0.15', '0.6', '-0.9']
    CHAR_My = ['0.', '0.15', '-0.5', '0.9']
    CHAR_Mz = ['0.', '0.4', '-1.2', '-1.6']
elif part.text == 'large':
    xml_msg = root.find('large')
    f_discriminant = discriminant_large
    # 第一批次数据，运行后保留：51959组，抛弃：2472组
    # CHAR_Fx = ['0.', '10.', '-30.', '60.', '-100.', '160.']
    # CHAR_Fy = ['0.', '-10.', '30.', '-60.', '100.', '-160.']
    # CHAR_Fz = ['0.', '-50.', '100.', '-200.', '400.', '-600.']
    # CHAR_Mx = ['0.', '200.', '-500.', '1000.', '-2500.', '4000.']
    # CHAR_My = ['0.', '-200.', '500.', '-1000.', '2500.', '-4000.']
    # CHAR_Mz = ['0.', '500.', '-1000.', '2500.', '-4000.', '6000.', '-9000.']
    # 第二批次数据，运行后保留：12319组，抛弃：180组
    # CHAR_Fx = ['0.', '-20.', '50.', '90.', '-130.']
    # CHAR_Fy = ['0.', '20.', '-40.', '-80.', '120.']
    # CHAR_Fz = ['0.', '-150.', '300.', '-400.', '500.']
    # CHAR_Mx = ['0.', '700.', '-1500.', '-2000.', '3000.']
    # CHAR_My = ['0.', '-800.', '1600.', '2000.', '-3000.']
    # CHAR_Mz = ['0.', '3000.', '-5000.', '8000.']
    # 第三批次数据，运行后保留：3987组，抛弃：108组
    CHAR_Fx = ['0.', '-45.', '-70.', '110.']
    CHAR_Fy = ['0.', '45.', '70.', '-110.']
    CHAR_Fz = ['0.', '-250.', '-350.', '450.']
    CHAR_Mx = ['0.', '-1200.', '2200.', '-3500.']
    CHAR_My = ['0.', '1200.', '-2200.', '3500.']
    CHAR_Mz = ['0.', '2000.', '-4500.', '7000.']
else:
    sys.exit('xml wrong! Please check <part>!')

# 读取文件保存路径
path_save = xml_msg.find('path_save').text.strip()

# 读取输出文件名
simu_list = xml_msg.find('FM/simu_list').text.strip()
simu_abandon = xml_msg.find('FM/simu_abandon').text.strip()

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 把力和力矩的选值进行组合，生成simu_list.csv和simu_abandon.csv
# simu_list.csv，记录各inp文件的具体的力和力矩值的组合
# simu_abandon.csv，记录被抛弃掉的力和力矩值的组合（可能发生屈服断裂的组合，不用仿真）

# 需要仿真的力和力矩值
# 将力和力矩值由字符串格式转化为浮点格式
len_Fx = len(CHAR_Fx)
len_Fy = len(CHAR_Fy)
len_Fz = len(CHAR_Fz)
len_Mx = len(CHAR_Mx)
len_My = len(CHAR_My)
len_Mz = len(CHAR_Mz)
NUM_Fx = [float(CHAR_Fx[i]) for i in range(len_Fx)]
NUM_Fy = [float(CHAR_Fy[i]) for i in range(len_Fy)]
NUM_Fz = [float(CHAR_Fz[i]) for i in range(len_Fz)]
NUM_Mx = [float(CHAR_Mx[i]) for i in range(len_Mx)]
NUM_My = [float(CHAR_My[i]) for i in range(len_My)]
NUM_Mz = [float(CHAR_Mz[i]) for i in range(len_Mz)]
# print(NUM_Fx, NUM_Fy, NUM_Fz, NUM_Mx, NUM_My, NUM_Mz)

path_simu_list = os.path.join(path_save, simu_list)
path_simu_abandon = os.path.join(path_save, simu_abandon)

# 以“写入”方式打开simu_list.csv
with open(path_simu_list, mode='w', encoding='utf8', newline='') as f_record:
    writer_record = csv.writer(f_record)

    # 以“写入”方式打开simu_abandon.csv
    with open(path_simu_abandon, mode='w', encoding='utf8', newline='') as f_abandon:
        writer_abandon = csv.writer(f_abandon)

        # 注意：simu_list.csv也是从0开始的，即第一行是Job-0，但是Job-0各力和力矩都是0，不需要仿真
        # 需要仿真的数据从Job-1开始，也就是从第二行开始
        # 即Job-n在simu_list.csv的第n+1行
        jobid = 0
        num_abandon = 0
        # 循环生成不同的力和力矩值的组合
        for ifx in range(len_Fx):
            for ify in range(len_Fy):
                for ifz in range(len_Fz):
                    for imx in range(len_Mx):
                        for imy in range(len_My):
                            for imz in range(len_Mz):
                                # 判断这种力和力矩值的组合是否有可能造成屈服断裂
                                # 很有可能屈服断裂，不用仿真
                                if not f_discriminant(NUM_Fx[ifx], NUM_Fy[ify], NUM_Fz[ifz], NUM_Mx[imx],
                                                      NUM_My[imy], NUM_Mz[imz]):
                                    char_abandon = [CHAR_Fx[ifx], CHAR_Fy[ify], CHAR_Fz[ifz], CHAR_Mx[imx],
                                                    CHAR_My[imy], CHAR_Mz[imz]]
                                    # 将抛弃的组合记录在simu_abandon.csv
                                    writer_abandon.writerow(char_abandon)
                                    print('A combination is abandoned! Due to the possibility of yielding.')
                                    # 统计抛弃的组合数，加一后的值表示已经在simu_abandon.csv写的行数
                                    num_abandon += 1
                                    continue
                                # 不太可能屈服断裂，需要仿真，生成新的inp文件
                                else:
                                    CHAR_JOBID = str(jobid)
                                    char_new = [CHAR_JOBID, CHAR_Fx[ifx], CHAR_Fy[ify], CHAR_Fz[ifz], CHAR_Mx[imx],
                                                CHAR_My[imy], CHAR_Mz[imz]]
                                    # 将新的inp文件的力和力矩的组合记录在simu_list.csv
                                    writer_record.writerow(char_new)
                                    print(jobid, ' is recorded.')
                                    # Job序列号加一，加一后的值表示已经在simu_list.csv写的行数
                                    jobid += 1
                                # 进入下一循环
        # 打印新生成的组合总数（但是要减去第一行的Job-0，因此是jobid - 1）
        print('Total jobs: ', jobid - 1)
        # 打印抛弃的组合数
        print('Abandoned combinations: ', num_abandon)

print('{} generated! : {}'.format(simu_list, path_simu_list))
print('{} generated! : {}'.format(simu_abandon, path_simu_abandon))

# ********************

print('\nCompleted!')

