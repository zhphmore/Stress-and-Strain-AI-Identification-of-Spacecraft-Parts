import os
from functools import partial
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import ui_first
import ui_small
import ui_large
import nn_small
import nn_large


# ********************************************************************************
# ****                                                                        ****
# ****                                 小型零件                                 ****
# ****                                                                        ****
# ********************************************************************************
class window_small(ui_small.Ui_Dialog, QDialog):
    def __init__(self):
        super(window_small, self).__init__()
        self.setupUi(self)

        # 窗口上方：去除问号，保留最小化、关闭
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        # ********************计算结果的储存********************
        # 储存各网格单元的mises应力
        self.data_S_mises = np.zeros(127105)
        # 储存最大mises应力对应的坐标
        self.max_S_mises_location = np.zeros(3)

        # ********************绘图初始化********************
        # 绘图区域，使用pyvista和pyqt的接口QtInteractor
        self.plotter = QtInteractor(self.frame_graph)
        self.plotter.set_background(color='#E3EFFC')
        self.plotter.show_axes()

        # 显示模式
        # 视图：默认网格视图
        self.comboBox.setCurrentIndex(0)
        # 显示应变片：默认显示
        self.radioButton.setChecked(True)

        # 绘制网格
        # 读取网格文件：各节点坐标、网格单元包含的节点的编号
        self.data_node_coordinate = pd.read_csv('node_coordinate_small.csv', header=None,
                                                encoding='utf-8').to_numpy()[:, 1:4]
        self.data_element_node = pd.read_csv('element_node_small.csv', header=None,
                                             encoding='utf-8').to_numpy()[:, 0:11]
        # 1个二阶四面体单元包含10个节点
        self.data_element_node[:, 0] = 10
        self.data_element_node[:, 1:11] -= 1
        # 设置网格单元的类型：二阶四面体单元
        self.data_celltypes = []
        num_elem = self.data_element_node.shape[0]
        for i in range(num_elem):
            self.data_celltypes.append(pv.CellType.QUADRATIC_TETRA)
        # 小型零件网格
        self.grid_small = pv.UnstructuredGrid(self.data_element_node, self.data_celltypes, self.data_node_coordinate)

        # 绘制应变片
        # 读取应变片相关文件
        self.data_clip_coordinate = pd.read_csv('clip_coordinate_small.csv', header=None,
                                                encoding='utf-8').to_numpy()[:, 1:4]
        self.data_clip_element = pd.read_csv('clip_element_small.csv', header=None,
                                             encoding='utf-8').to_numpy()[:, 0:9]
        # 1个一阶六面体单元包含8个节点
        self.data_clip_element[:, 0] = 8
        self.data_clip_element[:, 1:9] -= 1
        # 设置网格单元的类型：一阶六面体单元
        self.data_clip_celltypes = [pv.CellType.HEXAHEDRON]
        # 应变片网格
        self.grid_clip = []
        # 小型零件贴了6个应变片
        for i in range(6):
            self.grid_clip.append(pv.UnstructuredGrid(self.data_clip_element[i].reshape(1, 9), self.data_clip_celltypes,
                                                      self.data_clip_coordinate))

        # 各网格单元质心位置的坐标
        self.data_element_centroid = pd.read_csv('element_centroid_small.csv', header=None,
                                                 encoding='utf-8').to_numpy()[:, 1:4]
        # 各区域的网格单元的个数
        self.data_region_elem_ordered = pd.read_csv('region_elem_ordered_small.csv', header=None,
                                                    encoding="utf-8").to_numpy()

        # ********************初始绘图********************
        self.showPart()

        # ********************通过应变片应变辨识********************
        self.pushButton_11.clicked.connect(partial(self.showClip, 1))
        self.pushButton_12.clicked.connect(partial(self.showClip, 2))
        self.pushButton_13.clicked.connect(partial(self.showClip, 3))
        self.pushButton_14.clicked.connect(partial(self.showClip, 4))
        self.pushButton_15.clicked.connect(partial(self.showClip, 5))
        self.pushButton_16.clicked.connect(partial(self.showClip, 6))
        self.pushButton_101.clicked.connect(self.getDirectoryClip)
        self.pushButton_102.clicked.connect(self.readDirectoryClip)
        self.pushButton_103.clicked.connect(self.clearClip)
        self.pushButton_100.clicked.connect(self.doA)

        # ********************通过整体受力和力矩辨识********************
        self.pushButton_201.clicked.connect(self.getDirectoryFM)
        self.pushButton_202.clicked.connect(self.readDirectoryFM)
        self.pushButton_203.clicked.connect(self.clearFM)
        self.pushButton_200.clicked.connect(self.doB)

        # ********************导出辨识结果********************
        self.pushButton_301.clicked.connect(self.getDirectoryAns)
        self.pushButton_302.clicked.connect(self.writeDirectoryAns)

        # ********************绘图区域********************
        self.comboBox.currentIndexChanged.connect(self.showPart)
        self.radioButton.toggled.connect(self.showPart)
        self.pushButton_xy.clicked.connect(self.plotter.view_xy)
        self.pushButton_yx.clicked.connect(self.plotter.view_yx)
        self.pushButton_yz.clicked.connect(self.plotter.view_yz)
        self.pushButton_zy.clicked.connect(self.plotter.view_zy)
        self.pushButton_zx.clicked.connect(self.plotter.view_zx)
        self.pushButton_xz.clicked.connect(self.plotter.view_xz)
        # self.pushButton_isometric.clicked.connect(self.plotter.view_isometric)

        # ********************使用说明********************
        self.pushButton_00.clicked.connect(self.readmePart)
        self.pushButton_10.clicked.connect(self.readmeClip)
        self.pushButton_20.clicked.connect(self.readmeFM)
        self.pushButton_30.clicked.connect(self.readmeAns)

        # ********************关闭页面********************
        self.pushButton_0.clicked.connect(self.cancel)

    # ******************************应变片区域函数******************************
    # “选择文件”按钮：选择存放应变片应变的xml文件
    def getDirectoryClip(self):
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, caption='请选择存放应变片应变的xml文件',
                                                                     filter='xml (*.xml)')
        self.lineEdit_102.setText(file_path)

    # “读取”按钮：从xml文件读取应变片应变
    def readDirectoryClip(self):
        path_input = self.lineEdit_102.text().strip()
        if os.path.exists(path_input) and path_input.endswith('.xml'):
            root = ET.parse(path_input).getroot()
            xml_msg = root.find('small/clip')
            if xml_msg is not None:
                if xml_msg.find('S1') is not None:
                    self.lineEdit_11.setText(xml_msg.find('S1').text.strip())
                if xml_msg.find('S2') is not None:
                    self.lineEdit_12.setText(xml_msg.find('S2').text.strip())
                if xml_msg.find('S3') is not None:
                    self.lineEdit_13.setText(xml_msg.find('S3').text.strip())
                if xml_msg.find('S4') is not None:
                    self.lineEdit_14.setText(xml_msg.find('S4').text.strip())
                if xml_msg.find('S5') is not None:
                    self.lineEdit_15.setText(xml_msg.find('S5').text.strip())
                if xml_msg.find('S6') is not None:
                    self.lineEdit_16.setText(xml_msg.find('S6').text.strip())
            else:
                QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 未找到 <root> <small> <clip>')
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 不存在，请检查文件路径是否正确！')

    # “清空”按钮：清空应变片应变
    def clearClip(self):
        self.lineEdit_11.clear()
        self.lineEdit_12.clear()
        self.lineEdit_13.clear()
        self.lineEdit_14.clear()
        self.lineEdit_15.clear()
        self.lineEdit_16.clear()

    # 检查应变片应变格式是否正确，能否转化为数字
    def checkClip(self):
        try:
            data_clip = np.array([float(self.lineEdit_11.text()), float(self.lineEdit_12.text()),
                                  float(self.lineEdit_13.text()), float(self.lineEdit_14.text()),
                                  float(self.lineEdit_15.text()), float(self.lineEdit_16.text())])
            flag_input = True
        except ValueError:
            data_clip = np.array([0, 0, 0, 0, 0, 0])
            flag_input = False
        return flag_input, data_clip

    # “输出”按钮：根据应变片应变预测
    def doA(self):
        flag_input, data_clip = self.checkClip()
        if flag_input:
            data_FM = nn_small.nn_small_A(data_clip)
            self.setFM(data_FM)
            data_region_S, ans_IDSE = nn_small.nn_small_B(data_FM)
            for i in range(4087):
                self.data_S_mises[int(self.data_region_elem_ordered[i][1] - 1)] = data_region_S[i]
            self.showResult(data_FM, ans_IDSE)
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '应变片应变不是数字，格式错误')

    def setFM(self, data_FM):
        self.lineEdit_21.setText(str(data_FM[0]))
        self.lineEdit_22.setText(str(data_FM[1]))
        self.lineEdit_23.setText(str(data_FM[2]))
        self.lineEdit_24.setText(str(data_FM[3]))
        self.lineEdit_25.setText(str(data_FM[4]))
        self.lineEdit_26.setText(str(data_FM[5]))

    # ******************************力和力矩区域函数******************************
    def getDirectoryFM(self):
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, caption='请选择存放整体受力和力矩的xml文件',
                                                                     filter='xml (*.xml)')
        self.lineEdit_202.setText(file_path)

    def readDirectoryFM(self):
        path_input = self.lineEdit_202.text().strip()
        if os.path.exists(path_input) and path_input.endswith('.xml'):
            root = ET.parse(path_input).getroot()
            xml_msg = root.find('small/FM')
            if xml_msg is not None:
                if xml_msg.find('Fx') is not None:
                    self.lineEdit_21.setText(xml_msg.find('Fx').text.strip())
                if xml_msg.find('Fy') is not None:
                    self.lineEdit_22.setText(xml_msg.find('Fy').text.strip())
                if xml_msg.find('Fz') is not None:
                    self.lineEdit_23.setText(xml_msg.find('Fz').text.strip())
                if xml_msg.find('Mx') is not None:
                    self.lineEdit_24.setText(xml_msg.find('Mx').text.strip())
                if xml_msg.find('My') is not None:
                    self.lineEdit_25.setText(xml_msg.find('My').text.strip())
                if xml_msg.find('Mz') is not None:
                    self.lineEdit_26.setText(xml_msg.find('Mz').text.strip())
            else:
                QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 未找到 <root> <small> <FM>')
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 不存在，请检查文件路径是否正确！')

    def clearFM(self):
        self.lineEdit_21.clear()
        self.lineEdit_22.clear()
        self.lineEdit_23.clear()
        self.lineEdit_24.clear()
        self.lineEdit_25.clear()
        self.lineEdit_26.clear()

    # 检查整体受力和力矩格式是否正确，能否转化为数字
    def checkFM(self):
        try:
            data_FM = np.array([float(self.lineEdit_21.text()), float(self.lineEdit_22.text()),
                                float(self.lineEdit_23.text()), float(self.lineEdit_24.text()),
                                float(self.lineEdit_25.text()), float(self.lineEdit_26.text())])
            flag_input = True
        except ValueError:
            data_FM = np.array([0, 0, 0, 0, 0, 0])
            flag_input = False
        return flag_input, data_FM

    # “输出”按钮：根据整体受力和力矩预测
    def doB(self):
        flag_input, data_FM = self.checkFM()
        if flag_input:
            self.clearClip()
            data_region_S, ans_IDSE = nn_small.nn_small_B(data_FM)
            for i in range(4087):
                self.data_S_mises[int(self.data_region_elem_ordered[i][1] - 1)] = data_region_S[i]
            self.showResult(data_FM, ans_IDSE)
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '力和力矩不是数字，格式错误')

    def showResult(self, data_FM, ans_IDSE):
        text_11 = '小型零件整体受力:\nFx: {} N\nFy: {} N\nFz: {} N\n'.format(data_FM[0], data_FM[1], data_FM[2])
        text_12 = 'Mx: {} Nm\nMy: {} Nm\nMz: {} Nm\n'.format(data_FM[3], data_FM[4], data_FM[5])
        # ans_XYZ, node_target都是最危险位置的坐标，ans_XYZ是一维数组，node_target是二维数组，绘图需要二维数组
        ans_XYZ = self.data_element_centroid[int(ans_IDSE[0])]
        node_target = ans_XYZ.reshape((1, 3))
        text_2 = '最危险位置:\nx: {} mm\ny: {} mm\nz: {} mm\n'.format(ans_XYZ[0], ans_XYZ[1], ans_XYZ[2])
        text_31 = '最危险位置的 mises 应力:\nS_mises_max: {} MPa\n'.format(ans_IDSE[1])
        test_32 = '最危险位置的 应变分量:\nE11: {}\nE22: {}\nE33: {}\n'.format(ans_IDSE[2], ans_IDSE[3], ans_IDSE[4])
        test_33 = 'E12: {}\nE13: {}\nE23: {}\n'.format(ans_IDSE[5], ans_IDSE[6], ans_IDSE[7])
        text_msg = text_11 + text_12 + text_2 + text_31 + test_32 + test_33
        self.textBrowser.setText(text_msg)
        # 把视图调整为：mises 应力云图
        self.comboBox.setCurrentIndex(1)
        # 绘制零件
        self.showPart()
        # 绘制危险点
        self.plotter.add_mesh(pv.PolyData(node_target), color='red', point_size=10, render_points_as_spheres=True)

    # ******************************导出区域函数******************************
    def getDirectoryAns(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, caption='请选择结果导出到的文件夹')
        self.lineEdit_302.setText(folder_path)
        # 如果为空，则填写默认文件名 result_small.csv
        if not len(self.lineEdit_303.text().strip()):
            self.lineEdit_303.setText('result_small')

    def writeDirectoryAns(self):
        folder_output = self.lineEdit_302.text().strip()
        file_output_1 = self.lineEdit_303.text().strip() + '.csv'
        path_output_1 = os.path.join(folder_output, file_output_1)
        file_output_2 = self.lineEdit_303.text().strip() + '_info.txt'
        path_output_2 = os.path.join(folder_output, file_output_2)
        flag_output = False
        if os.path.isdir(folder_output):
            # 默认关闭界面选择No
            if os.path.exists(path_output_1):
                if QMessageBox.Yes == QMessageBox.question(self, '温馨提示', '该文件已存在，是否覆盖？',
                                                           QMessageBox.Yes | QMessageBox.No,
                                                           QMessageBox.No):
                    flag_output = True
            else:
                flag_output = True
        else:
            QMessageBox.critical(None, '温馨提示', '文件夹不存在，请重新选择文件夹！')
        if flag_output:
            # 导出第一份文件 result_small.csv
            with open(path_output_1, mode='w', encoding='utf8', newline='') as f_output_1:
                # writer_output_1 = csv.writer(f_output_1)
                f_output_1.write('x/mm,y/mm,z/mm,S_mises\n')
                for i in range(4087):
                    elem_i = int(self.data_region_elem_ordered[i][1])
                    j = int(elem_i - 1)
                    row_str_i = str(self.data_element_centroid[j][0]) + ',' + str(
                        self.data_element_centroid[j][1]) + ',' + str(
                        self.data_element_centroid[j][2]) + ',' + str(
                        self.data_S_mises[j]) + '\n'
                    f_output_1.write(row_str_i)
            # 导出第二份文件 result_small_info.txt
            with open(path_output_2, mode='w', encoding='utf8', newline='') as f_output_2:
                f_output_2.write(self.textBrowser.toPlainText())
            QMessageBox.information(None, '温馨提示', '结果文件已成功导出！')

    # ******************************绘图区域函数******************************
    # 绘图展示应变片的位置
    def showClip(self, clip_id):
        if not 1 <= int(clip_id) <= 6:
            return
        self.radioButton.setChecked(True)
        self.plotter.clear_actors()
        # 视图：网格视图
        if int(self.comboBox.currentIndex()) == 0:
            i = int(clip_id - 1)
            self.plotter.add_mesh(self.grid_small, name='smallpart', color=pv.global_theme.color,
                                  show_edges=True, line_width=1, show_scalar_bar=False)
            self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen', show_edges=True,
                                  line_width=10)
        # 视图：mises 应力云图
        elif int(self.comboBox.currentIndex()) == 1:
            i = int(clip_id - 1)
            self.plotter.add_mesh(self.grid_small, name='smallpart', scalars=self.data_S_mises, cmap='jet',
                                  show_edges=True, line_width=1, show_scalar_bar=True,
                                  scalar_bar_args={'title': "S mises(MPa)", 'color': 'firebrick', 'title_font_size': 15,
                                                   'label_font_size': 12, 'width': 0.5,
                                                   'vertical': False})
            self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen', show_edges=True,
                                  line_width=10)
        # 调整摄像机位置
        if int(clip_id) == 1 or int(clip_id) == 2:
            self.plotter.view_yz()
        elif int(clip_id) == 3:
            self.plotter.view_zy()
        elif int(clip_id) == 4 or int(clip_id) == 5:
            self.plotter.view_zx()
        elif int(clip_id) == 6:
            self.plotter.view_xz()

    # 绘图展示零件
    def showPart(self):
        self.plotter.clear_actors()
        # 视图：网格视图
        if int(self.comboBox.currentIndex()) == 0:
            self.plotter.add_mesh(self.grid_small, name='smallpart', color=pv.global_theme.color,
                                  show_edges=True, line_width=1, show_scalar_bar=False)
            # 显示应变片：显示
            if self.radioButton.isChecked():
                for i in range(6):
                    self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen',
                                          show_edges=True, line_width=10)
        # 视图：mises 应力云图
        elif int(self.comboBox.currentIndex()) == 1:
            self.plotter.add_mesh(self.grid_small, name='smallpart', scalars=self.data_S_mises, cmap='jet',
                                  show_edges=True, line_width=1, show_scalar_bar=True,
                                  scalar_bar_args={'title': "S mises(MPa)", 'color': 'firebrick', 'title_font_size': 15,
                                                   'label_font_size': 12, 'width': 0.5,
                                                   'vertical': False})
            # 显示应变片：显示
            if self.radioButton.isChecked():
                for i in range(6):
                    self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen',
                                          show_edges=True, line_width=10)

    # ******************************使用说明函数******************************
    def readmePart(self):
        txt_readme = ('小型零件    基本信息\n\n'
                      '尺寸：\n'
                      '外半径：8mm，内半径：4mm，高度：32mm\n\n'
                      '材料：\n'
                      'TC4钛合金（Ti-6Al-4V）\n'
                      '弹性：模量 E = 110GPa，泊松比 υ = 0.34\n'
                      '塑性：Johnson-Cook模型：\n'
                      'A = 1060 MPa，B = 1090 MPa，n = 0.884，m = 1.1\n'
                      'T_emit = 1878 K，T_r = 293 K\n\n'
                      '网格：\n'
                      '四面体网格单元 C3D10，布种间距：0.8 mm\n'
                      '单元数：127105，节点数：188191\n\n'
                      '特别说明：\n'
                      'mises应力：单元应力（单元质心位置处应力，并非节点应力）\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    def readmeClip(self):
        txt_readme = ('小型零件    通过应变片应变辨识\n'
                      '使用说明：\n\n'
                      '第一步：在6个应变片输入框输入各应变片应变\n'
                      '第二步：点击“输出”，稍等几秒，自动依次计算：\n'
                      '1、对应外载荷填充下方\n'
                      '2、根据上述外载荷计算关键区域各点mises应力并绘制云图\n'
                      '3、应力最大值、位置及应变分量\n\n'
                      '支持两种输入方式：\n'
                      '方法一： 直接在6个应变片输入框输入（支持科学计数法：例如：0.2e-5）\n'
                      '方法二： “选择文件”选择xml文件，“读取”自动填充输入框\n'
                      '“清空”：清除6个应变片输入框内容\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    def readmeFM(self):
        txt_readme = ('小型零件    通过整体受力和力矩辨识\n'
                      '使用说明：\n\n'
                      '第一步：在6个外载荷输入框输入各方向力和力矩\n'
                      '第二步：点击“输出”，稍等几秒，自动依次计算：\n'
                      '1、关键区域各点mises应力并绘制云图\n'
                      '2、应力最大值、位置及应变分量\n\n'
                      '支持两种输入方式：\n'
                      '方法一： 直接在6个外载荷输入框输入（支持科学计数法）\n'
                      '方法二： “选择文件”选择xml文件，“读取”自动填充输入框\n'
                      '“清空”：清除6个应变片输入框内容\n\n'
                      '特别说明：\n'
                      '零件底面（z=0mm）固定，顶面（z=32mm）承受载荷\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    def readmeAns(self):
        txt_readme = ('小型零件    导出辨识结果\n'
                      '使用说明：\n\n'
                      '第一步：“选择文件”选择导出位置\n'
                      '第二步：输入导出的文件名 xxxx\n'
                      '第三步：点击“导出”，自动导出：\n'
                      '1、1个csv文件，xxxx.csv，关键区域各点mises应力\n'
                      '2、1个txt文件，xxxx_info.txt，内容为界面右下方信息框内容\n'
                      '（应力最大值、位置及应变分量）\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    # 返回按钮
    def cancel(self):
        # self.close() 会调用 self.closeEvent()
        self.close()

    # 重写关闭事件
    def closeEvent(self, event):
        # 必要：关闭绘图工具！
        self.plotter.close()
        self.clearClip()
        self.clearFM()
        event.accept()


# ********************************************************************************
# ****                                                                        ****
# ****                                 大型零件                                 ****
# ****                                                                        ****
# ********************************************************************************
class window_large(ui_large.Ui_Dialog, QDialog):
    def __init__(self):
        super(window_large, self).__init__()
        self.setupUi(self)

        # 窗口上方：去除问号，保留最小化、关闭
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        # ********************计算结果的储存********************
        # 储存各网格单元的mises应力
        self.data_S_mises = np.zeros(194447)
        # 储存最大mises应力对应的坐标
        self.max_S_mises_location = np.zeros(3)

        # ********************绘图初始化********************
        # 绘图区域，使用pyvista和pyqt的接口QtInteractor
        self.plotter = QtInteractor(self.frame_graph)
        self.plotter.set_background(color='#E3EFFC')
        self.plotter.show_axes()

        # 显示模式
        # 视图：默认网格视图
        self.comboBox.setCurrentIndex(0)
        # 显示应变片：默认显示
        self.radioButton.setChecked(True)

        # 绘制网格
        # 读取网格文件：各节点坐标、网格单元包含的节点的编号
        self.data_node_coordinate = pd.read_csv('node_coordinate_large.csv', header=None,
                                                encoding='utf-8').to_numpy()[:, 1:4]
        self.data_element_node = pd.read_csv('element_node_large.csv', header=None,
                                             encoding='utf-8').to_numpy()[:, 0:11]
        self.data_element_centroid = pd.read_csv('element_centroid_large.csv', header=None,
                                                 encoding='utf-8').to_numpy()[:, 1:4]
        # 1个二阶四面体单元包含10个节点
        self.data_element_node[:, 0] = 10
        self.data_element_node[:, 1:11] -= 1
        # 设置网格单元的类型：二阶四面体单元
        self.data_celltypes = []
        num_elem = self.data_element_node.shape[0]
        for i in range(num_elem):
            self.data_celltypes.append(pv.CellType.QUADRATIC_TETRA)
        # 大型零件网格
        self.grid_large = pv.UnstructuredGrid(self.data_element_node, self.data_celltypes, self.data_node_coordinate)

        # 绘制应变片
        # 读取应变片相关文件
        self.data_clip_coordinate = pd.read_csv('clip_coordinate_large.csv', header=None,
                                                encoding='utf-8').to_numpy()[:, 1:4]
        self.data_clip_element = pd.read_csv('clip_element_large.csv', header=None,
                                             encoding='utf-8').to_numpy()[:, 0:9]
        # 1个一阶六面体单元包含8个节点
        self.data_clip_element[:, 0] = 8
        self.data_clip_element[:, 1:9] -= 1
        # 设置网格单元的类型：一阶六面体单元
        self.data_clip_celltypes = [pv.CellType.HEXAHEDRON]
        # 应变片网格
        self.grid_clip = []
        # 大型零件贴了8个应变片
        for i in range(8):
            self.grid_clip.append(pv.UnstructuredGrid(self.data_clip_element[i].reshape(1, 9), self.data_clip_celltypes,
                                                      self.data_clip_coordinate))

        # 各网格单元质心位置的坐标
        self.data_element_centroid = pd.read_csv('element_centroid_large.csv', header=None,
                                                 encoding='utf-8').to_numpy()[:, 1:4]
        # 各区域的网格单元的个数
        self.data_region_elem_ordered = pd.read_csv('region_elem_ordered_large.csv', header=None,
                                                    encoding="utf-8").to_numpy()

        # ********************初始绘图********************
        self.showPart()

        # ********************通过应变片应变辨识********************
        self.pushButton_11.clicked.connect(partial(self.showClip, 1))
        self.pushButton_12.clicked.connect(partial(self.showClip, 2))
        self.pushButton_13.clicked.connect(partial(self.showClip, 3))
        self.pushButton_14.clicked.connect(partial(self.showClip, 4))
        self.pushButton_15.clicked.connect(partial(self.showClip, 5))
        self.pushButton_16.clicked.connect(partial(self.showClip, 6))
        self.pushButton_17.clicked.connect(partial(self.showClip, 7))
        self.pushButton_18.clicked.connect(partial(self.showClip, 8))
        self.pushButton_101.clicked.connect(self.getDirectoryClip)
        self.pushButton_102.clicked.connect(self.readDirectoryClip)
        self.pushButton_103.clicked.connect(self.clearClip)
        self.pushButton_100.clicked.connect(self.doA)

        # ********************通过整体受力和力矩辨识********************
        self.pushButton_201.clicked.connect(self.getDirectoryFM)
        self.pushButton_202.clicked.connect(self.readDirectoryFM)
        self.pushButton_203.clicked.connect(self.clearFM)
        self.pushButton_200.clicked.connect(self.doB)

        # ********************导出辨识结果********************
        self.pushButton_301.clicked.connect(self.getDirectoryAns)
        self.pushButton_302.clicked.connect(self.writeDirectoryAns)

        # ********************绘图区域********************
        self.comboBox.currentIndexChanged.connect(self.showPart)
        self.radioButton.toggled.connect(self.showPart)
        self.pushButton_xy.clicked.connect(self.plotter.view_xy)
        self.pushButton_yx.clicked.connect(self.plotter.view_yx)
        self.pushButton_yz.clicked.connect(self.plotter.view_yz)
        self.pushButton_zy.clicked.connect(self.plotter.view_zy)
        self.pushButton_zx.clicked.connect(self.plotter.view_zx)
        self.pushButton_xz.clicked.connect(self.plotter.view_xz)
        # self.pushButton_isometric.clicked.connect(self.plotter.view_isometric)

        # ********************使用说明********************
        self.pushButton_00.clicked.connect(self.readmePart)
        self.pushButton_10.clicked.connect(self.readmeClip)
        self.pushButton_20.clicked.connect(self.readmeFM)
        self.pushButton_30.clicked.connect(self.readmeAns)

        # ********************关闭页面********************
        self.pushButton_0.clicked.connect(self.cancel)

    # ******************************应变片区域函数******************************
    # “选择文件”按钮：选择存放应变片应变的xml文件
    def getDirectoryClip(self):
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, caption='请选择存放应变片应变的xml文件',
                                                                     filter='xml (*.xml)')
        self.lineEdit_102.setText(file_path)

    # “读取”按钮：从xml文件读取应变片应变
    def readDirectoryClip(self):
        path_input = self.lineEdit_102.text().strip()
        if os.path.exists(path_input) and path_input.endswith('.xml'):
            root = ET.parse(path_input).getroot()
            xml_msg = root.find('large/clip')
            if xml_msg is not None:
                if xml_msg.find('S1') is not None:
                    self.lineEdit_11.setText(xml_msg.find('S1').text.strip())
                if xml_msg.find('S2') is not None:
                    self.lineEdit_12.setText(xml_msg.find('S2').text.strip())
                if xml_msg.find('S3') is not None:
                    self.lineEdit_13.setText(xml_msg.find('S3').text.strip())
                if xml_msg.find('S4') is not None:
                    self.lineEdit_14.setText(xml_msg.find('S4').text.strip())
                if xml_msg.find('S5') is not None:
                    self.lineEdit_15.setText(xml_msg.find('S5').text.strip())
                if xml_msg.find('S6') is not None:
                    self.lineEdit_16.setText(xml_msg.find('S6').text.strip())
                if xml_msg.find('S7') is not None:
                    self.lineEdit_17.setText(xml_msg.find('S7').text.strip())
                if xml_msg.find('S8') is not None:
                    self.lineEdit_18.setText(xml_msg.find('S8').text.strip())
            else:
                QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 未找到 <root> <large> <clip>')
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 不存在，请检查文件路径是否正确！')

    # “清空”按钮：清空应变片应变
    def clearClip(self):
        self.lineEdit_11.clear()
        self.lineEdit_12.clear()
        self.lineEdit_13.clear()
        self.lineEdit_14.clear()
        self.lineEdit_15.clear()
        self.lineEdit_16.clear()
        self.lineEdit_17.clear()
        self.lineEdit_18.clear()

    # 检查应变片应变格式是否正确，能否转化为数字
    def checkClip(self):
        try:
            data_clip = np.array([float(self.lineEdit_11.text()), float(self.lineEdit_12.text()),
                                  float(self.lineEdit_13.text()), float(self.lineEdit_14.text()),
                                  float(self.lineEdit_15.text()), float(self.lineEdit_16.text()),
                                  float(self.lineEdit_17.text()), float(self.lineEdit_18.text())])
            flag_input = True
        except ValueError:
            data_clip = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            flag_input = False
        return flag_input, data_clip

    # “输出”按钮：根据应变片应变预测
    def doA(self):
        flag_input, data_clip = self.checkClip()
        if flag_input:
            data_FM = nn_large.nn_large_A(data_clip)
            self.setFM(data_FM)
            data_region_S, ans_IDSE = nn_large.nn_large_B(data_FM)
            for i in range(4693):
                self.data_S_mises[int(self.data_region_elem_ordered[i][1] - 1)] = data_region_S[i]
            self.showResult(data_FM, ans_IDSE)
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '应变片应变不是数字，格式错误')

    def setFM(self, data_FM):
        self.lineEdit_21.setText(str(data_FM[0]))
        self.lineEdit_22.setText(str(data_FM[1]))
        self.lineEdit_23.setText(str(data_FM[2]))
        self.lineEdit_24.setText(str(data_FM[3]))
        self.lineEdit_25.setText(str(data_FM[4]))
        self.lineEdit_26.setText(str(data_FM[5]))

    # ******************************力和力矩区域函数******************************
    def getDirectoryFM(self):
        file_path, file_type = QtWidgets.QFileDialog.getOpenFileName(self, caption='请选择存放整体受力和力矩的xml文件',
                                                                     filter='xml (*.xml)')
        self.lineEdit_202.setText(file_path)

    def readDirectoryFM(self):
        path_input = self.lineEdit_202.text().strip()
        if os.path.exists(path_input) and path_input.endswith('.xml'):
            root = ET.parse(path_input).getroot()
            xml_msg = root.find('large/FM')
            if xml_msg is not None:
                if xml_msg.find('Fx') is not None:
                    self.lineEdit_21.setText(xml_msg.find('Fx').text.strip())
                if xml_msg.find('Fy') is not None:
                    self.lineEdit_22.setText(xml_msg.find('Fy').text.strip())
                if xml_msg.find('Fz') is not None:
                    self.lineEdit_23.setText(xml_msg.find('Fz').text.strip())
                if xml_msg.find('Mx') is not None:
                    self.lineEdit_24.setText(xml_msg.find('Mx').text.strip())
                if xml_msg.find('My') is not None:
                    self.lineEdit_25.setText(xml_msg.find('My').text.strip())
                if xml_msg.find('Mz') is not None:
                    self.lineEdit_26.setText(xml_msg.find('Mz').text.strip())
            else:
                QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 未找到 <root> <large> <FM>')
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', 'xml 不存在，请检查文件路径是否正确！')

    def clearFM(self):
        self.lineEdit_21.clear()
        self.lineEdit_22.clear()
        self.lineEdit_23.clear()
        self.lineEdit_24.clear()
        self.lineEdit_25.clear()
        self.lineEdit_26.clear()

    # 检查整体受力和力矩格式是否正确，能否转化为数字
    def checkFM(self):
        try:
            data_FM = np.array([float(self.lineEdit_21.text()), float(self.lineEdit_22.text()),
                                float(self.lineEdit_23.text()), float(self.lineEdit_24.text()),
                                float(self.lineEdit_25.text()), float(self.lineEdit_26.text())])
            flag_input = True
        except ValueError:
            data_FM = np.array([0, 0, 0, 0, 0, 0])
            flag_input = False
        return flag_input, data_FM

    # “输出”按钮：根据整体受力和力矩预测
    def doB(self):
        flag_input, data_FM = self.checkFM()
        if flag_input:
            self.clearClip()
            data_region_S, ans_IDSE = nn_large.nn_large_B(data_FM)
            for i in range(4693):
                self.data_S_mises[int(self.data_region_elem_ordered[i][1] - 1)] = data_region_S[i]
            self.showResult(data_FM, ans_IDSE)
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '力和力矩不是数字，格式错误')

    def showResult(self, data_FM, ans_IDSE):
        text_11 = '大型零件整体受力:\nFx: {} N\nFy: {} N\nFz: {} N\n'.format(data_FM[0], data_FM[1], data_FM[2])
        text_12 = 'Mx: {} Nm\nMy: {} Nm\nMz: {} Nm\n'.format(data_FM[3], data_FM[4], data_FM[5])
        ans_XYZ = self.data_element_centroid[int(ans_IDSE[0])]
        node_target = np.zeros((1, 3))
        node_target[0, :] = ans_XYZ
        text_2 = '最危险位置:\nx: {} mm\ny: {} mm\nz: {} mm\n'.format(ans_XYZ[0], ans_XYZ[1], ans_XYZ[2])
        text_31 = '最危险位置的 mises 应力:\nS_mises_max: {} MPa\n'.format(ans_IDSE[1])
        test_32 = '最危险位置的 应变分量（微应变）:\nE11: {}\nE22: {}\nE33: {}\n'.format(ans_IDSE[2], ans_IDSE[3],
                                                                                       ans_IDSE[4])
        test_33 = 'E12: {}\nE13: {}\nE23: {}\n'.format(ans_IDSE[5], ans_IDSE[6], ans_IDSE[7])
        text_msg = text_11 + text_12 + text_2 + text_31 + test_32 + test_33
        self.textBrowser.setText(text_msg)
        self.comboBox.setCurrentIndex(1)
        self.showPart()
        self.plotter.add_mesh(pv.PolyData(node_target), color='red', point_size=10, render_points_as_spheres=True)

    # ******************************导出区域函数******************************
    def getDirectoryAns(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, caption='请选择结果导出到的文件夹')
        self.lineEdit_302.setText(folder_path)
        # 如果为空，则填写默认文件名 result_large.csv
        if not len(self.lineEdit_303.text().strip()):
            self.lineEdit_303.setText('result_large')

    def writeDirectoryAns(self):
        folder_output = self.lineEdit_302.text().strip()
        file_output_1 = self.lineEdit_303.text().strip() + '.csv'
        path_output_1 = os.path.join(folder_output, file_output_1)
        file_output_2 = self.lineEdit_303.text().strip() + '_info.txt'
        path_output_2 = os.path.join(folder_output, file_output_2)
        flag_output = False
        if os.path.isdir(folder_output):
            # 默认关闭界面选择No
            if os.path.exists(path_output_1):
                if QMessageBox.Yes == QMessageBox.question(self, '温馨提示', '该文件已存在，是否覆盖？',
                                                           QMessageBox.Yes | QMessageBox.No,
                                                           QMessageBox.No):
                    flag_output = True
            else:
                flag_output = True
        else:
            QMessageBox.critical(None, '温馨提示', '文件夹不存在，请重新选择文件夹！')
        if flag_output:
            # 导出第一份文件 result_large.csv
            with open(path_output_1, mode='w', encoding='utf8', newline='') as f_output_1:
                # writer_output_1 = csv.writer(f_output_1)
                f_output_1.write('x/mm,y/mm,z/mm,S_mises\n')
                for i in range(4693):
                    elem_i = int(self.data_region_elem_ordered[i][1])
                    j = int(elem_i - 1)
                    row_str_i = str(self.data_element_centroid[j][0]) + ',' + str(
                        self.data_element_centroid[j][1]) + ',' + str(
                        self.data_element_centroid[j][2]) + ',' + str(
                        self.data_S_mises[j]) + '\n'
                    f_output_1.write(row_str_i)
            # 导出第二份文件 result_large_info.txt
            with open(path_output_2, mode='w', encoding='utf8', newline='') as f_output_2:
                f_output_2.write(self.textBrowser.toPlainText())
            QMessageBox.information(None, '温馨提示', '结果文件已成功导出！')

    # ******************************绘图区域函数******************************
    # 绘图展示应变片的位置
    def showClip(self, clip_id):
        if not 1 <= int(clip_id) <= 8:
            return
        self.radioButton.setChecked(True)
        self.plotter.clear_actors()
        # 视图：网格视图
        if int(self.comboBox.currentIndex()) == 0:
            i = int(clip_id - 1)
            self.plotter.add_mesh(self.grid_large, name='largepart', color=pv.global_theme.color,
                                  show_edges=True, line_width=1, show_scalar_bar=False)
            self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen', show_edges=True,
                                  line_width=10)
        # 视图：mises 应力云图
        elif int(self.comboBox.currentIndex()) == 1:
            i = int(clip_id - 1)
            self.plotter.add_mesh(self.grid_large, name='largepart', scalars=self.data_S_mises, cmap='jet',
                                  show_edges=True, line_width=1,
                                  scalar_bar_args={'title': "S mises(MPa)", 'color': 'firebrick', 'title_font_size': 15,
                                                   'label_font_size': 12, 'width': 0.5,
                                                   'vertical': False})
            self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen', show_edges=True,
                                  line_width=10)
        # 调整摄像机位置
        if 1 <= int(clip_id) <= 4:
            self.plotter.view_yz()
        elif 5 <= int(clip_id) <= 8:
            self.plotter.view_zx()

    # 绘图展示零件
    def showPart(self):
        self.plotter.clear_actors()
        # 视图：网格视图
        if int(self.comboBox.currentIndex()) == 0:
            self.plotter.add_mesh(self.grid_large, name='largepart', color=pv.global_theme.color,
                                  show_edges=True, line_width=1, show_scalar_bar=False)
            # 显示应变片：显示
            if self.radioButton.isChecked():
                for i in range(8):
                    self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen',
                                          show_edges=True, line_width=10)
        # 视图：mises 应力云图
        elif int(self.comboBox.currentIndex()) == 1:
            self.plotter.add_mesh(self.grid_large, name='largepart', scalars=self.data_S_mises, cmap='jet',
                                  show_edges=True, line_width=1,
                                  scalar_bar_args={'title': "S mises(MPa)", 'color': 'firebrick', 'title_font_size': 15,
                                                   'label_font_size': 12, 'width': 0.5,
                                                   'vertical': False})
            # 显示应变片：显示
            if self.radioButton.isChecked():
                for i in range(8):
                    self.plotter.add_mesh(self.grid_clip[i], color='yellowgreen', edge_color='yellowgreen',
                                          show_edges=True, line_width=10)

    # ******************************使用说明函数******************************
    def readmePart(self):
        txt_readme = ('大型零件    基本信息\n\n'
                      '尺寸：\n'
                      '外半径：15mm，内半径：10mm，高度：55mm\n\n'
                      '材料：\n'
                      'TC4钛合金（Ti-6Al-4V）\n'
                      '弹性：模量 E = 110GPa，泊松比 υ = 0.34\n'
                      '塑性：Johnson-Cook模型：\n'
                      'A = 1060 MPa，B = 1090 MPa，n = 0.884，m = 1.1\n'
                      'T_emit = 1878 K，T_r = 293 K\n\n'
                      '网格：\n'
                      '四面体网格单元 C3D10，布种间距：1 mm\n'
                      '单元数：194447，节点数：290225\n\n'
                      '特别说明：\n'
                      'mises应力：单元应力（单元质心位置处应力，并非节点应力）\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    def readmeClip(self):
        txt_readme = ('大型零件    通过应变片应变辨识\n'
                      '使用说明：\n\n'
                      '第一步：在8个应变片输入框输入各应变片应变（微应变）\n'
                      '第二步：点击“输出”，稍等几秒，自动依次计算：\n'
                      '1、对应外载荷填充下方\n'
                      '2、根据上述外载荷计算关键区域各点mises应力并绘制云图\n'
                      '3、应力最大值、位置及应变分量\n\n'
                      '支持两种输入方式：\n'
                      '方法一： 直接在8个应变片输入框输入（支持科学计数法：例如：0.2e-5）\n'
                      '方法二： “选择文件”选择xml文件，“读取”自动填充输入框\n'
                      '“清空”：清除8个应变片输入框内容\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    def readmeFM(self):
        txt_readme = ('大型零件    通过整体受力和力矩辨识\n'
                      '使用说明：\n\n'
                      '第一步：在6个外载荷输入框输入各方向力和力矩\n'
                      '第二步：点击“输出”，稍等几秒，自动依次计算：\n'
                      '1、关键区域各点mises应力并绘制云图\n'
                      '2、应力最大值、位置及应变分量\n\n'
                      '支持两种输入方式：\n'
                      '方法一： 直接在6个外载荷输入框输入（支持科学计数法）\n'
                      '方法二： “选择文件”选择xml文件，“读取”自动填充输入框\n'
                      '“清空”：清除8个应变片输入框内容\n\n'
                      '特别说明：\n'
                      '零件底面（z=0mm）固定，顶面（z=55mm）承受载荷\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    def readmeAns(self):
        txt_readme = ('大型零件    导出辨识结果\n'
                      '使用说明：\n\n'
                      '第一步：“选择文件”选择导出位置\n'
                      '第二步：输入导出的文件名 xxxx\n'
                      '第三步：点击“导出”，自动导出：\n'
                      '1、1个csv文件，xxxx.csv，关键区域各点mises应力\n'
                      '2、1个txt文件，xxxx_info.txt，内容为界面右下方信息框内容\n'
                      '（应力最大值、位置及应变分量）\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    # ******************************关闭事件******************************
    # 返回按钮
    def cancel(self):
        # self.close() 会调用 self.closeEvent()
        self.close()

    # 重写关闭事件
    def closeEvent(self, event):
        # 必要：关闭绘图工具！
        self.plotter.close()
        self.clearClip()
        self.clearFM()
        event.accept()


# ********************************************************************************
# ****                                                                        ****
# ****                                 主界面                                  ****
# ****                                                                        ****
# ********************************************************************************
class window_first(ui_first.Ui_Dialog, QDialog):
    def __init__(self):
        super(window_first, self).__init__()
        self.setupUi(self)

        # 窗口上方：去除问号，保留最小化、关闭
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        # ********************设置按钮********************
        self.pushButton_1.clicked.connect(self.runSmall)
        self.pushButton_2.clicked.connect(self.runLarge)

    def runSmall(self):
        dialog_small = window_small()
        dialog_small.show()

    def runLarge(self):
        dialog_large = window_large()
        dialog_large.show()
