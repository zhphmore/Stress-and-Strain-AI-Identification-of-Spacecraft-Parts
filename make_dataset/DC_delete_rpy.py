import os

# ******************************************************************************** #
# ******************************************************************************** #
# DC_delete_rpy.py
# 运行完DB.bat，也就是重复并行运行完odb_preserve.py后
# 会剩下一系列rpy文件
# 这些rpy文件是不需要的，都删了吧
# 程序DC的作用，就是删除当前文件夹下所有rpy文件
# ******************************************************************************** #
# ******************************************************************************** #

# **************************************** #
# 执行程序
# **************************************** #
# ********************
# 删除文件夹内不是odb的文件
# 待处理文件夹位置，默认为当前文件夹，可在此处修改为其它文件夹
path_current = os.path.dirname(os.path.realpath(__file__))
path_folder = path_current

# 需要删除的文件的后缀
str_start = 'abaqus.rpy'

cnt = 0
# 删除abaqus.rpy文件，并统计删除了多少个文件
for root, dirs, files in os.walk(path_folder):
    for file in files:
        # 检查每一个文件的后缀
        file_check = os.path.join(root, file)
        if file.startswith(str_start):
            os.remove(file_check)
            cnt += 1

print('Delete {} files'.format(cnt))

# ********************

print('\nCompleted!')
