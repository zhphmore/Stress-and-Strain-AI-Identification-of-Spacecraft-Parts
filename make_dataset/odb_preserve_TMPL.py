from abaqus import *
from abaqusConstants import *
from odbAccess import openOdb
from textRepr import *
import sys
import os
import csv

folder_save = 'TBDSAVEFOLDER'
element_preserve = 'TBDELEMPRSV'
folder_odb = 'TBDODBFOLDER'
folder_data = 'TBDDATAFOLDER'
file_data_name = 'TBDDATAFILE'

path_element_preserve = os.path.join(folder_save, element_preserve)
elem_Labels = []
f_element_preserve = open(path_element_preserve, 'r')
data_element_preserve = csv.reader(f_element_preserve)
for row in data_element_preserve:
    elem_Labels.append(int(row[0]))
f_element_preserve.close()

job_id = sys.argv[8]
str_num = str(job_id)

file_odb = 'Job-' + str_num + '.odb'
file_data = str_num + file_data_name

path_odb = os.path.join(folder_odb, file_odb)
path_data = os.path.join(folder_data, file_data)

my_odb = openOdb(path_odb)

values_S = my_odb.steps['Step-1'].frames[-1].fieldOutputs['S'].getSubset(position=CENTROID)
values_E = my_odb.steps['Step-1'].frames[-1].fieldOutputs['E'].getSubset(position=CENTROID)
values_Y = my_odb.steps['Step-1'].frames[-1].fieldOutputs['AC YIELD'].getSubset(position=CENTROID)

ff = open(path_data, 'w')
str_csv_title_S = 'elementLabel_CENTROID,S11,S22,S33,S12,S13,S23,S_maxPrincipal,S_midPrincipal,S_minPrincipal,S_mises,'
str_csv_title_E = 'E11,E22,E33,E12,E13,E23,E_maxPrincipal,E_midPrincipal,E_minPrincipal,E_mises,'
str_csv_title_Y = 'AC_YIELD_data\n'
str_csv_title = str_csv_title_S + str_csv_title_E + str_csv_title_Y
ff.write(str_csv_title)
for elem_label in elem_Labels:
    row_id = int(elem_label - 1)
    val_S = values_S.values[row_id]
    val_E = values_E.values[row_id]
    val_Y = values_Y.values[row_id]
    csv_line_S = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, '.format(val_S.elementLabel, val_S.data[0],
                                                                       val_S.data[1], val_S.data[2], val_S.data[3],
                                                                       val_S.data[4], val_S.data[5],
                                                                       val_S.maxPrincipal, val_S.midPrincipal,
                                                                       val_S.minPrincipal, val_S.mises)
    csv_line_E = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, '.format(val_E.data[0], val_E.data[1], val_E.data[2],
                                                                   val_E.data[3], val_E.data[4], val_E.data[5],
                                                                   val_E.maxPrincipal, val_E.midPrincipal,
                                                                   val_E.minPrincipal, val_E.mises)
    csv_line_Y = '{}\n'.format(val_Y.data)
    csv_line = csv_line_S + csv_line_E + csv_line_Y
    ff.write(csv_line)
ff.close()
my_odb.close()
