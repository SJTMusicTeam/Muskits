import numpy as np
import os
import shutil

file_dir = "PJS_corpus_ver1.1_new/mono_label/"
file_dir_new = "PJS_corpus_ver1.1_new/mono_label_new/"
os.mkdir("PJS_corpus_ver1.1_new/mono_label_new")
for file_index in range(1, 101):

    if (file_index <= 9):
        file_name_index = "0" + str(file_index)
    else:
        file_name_index = str(file_index)

    if file_index < 100:
        with open(file_dir + "pjs0" + file_name_index + ".lab", 'r') as f:
            file = f.read()
    else:
        with open(file_dir + "pjs" + file_name_index + ".lab", 'r') as f:
            file = f.read()

    lineList = file.split('\n')

    lineListLen = len(lineList)
    elementList = []
    for i in range(lineListLen - 1):
        elementList.append(lineList[i].split(' '))

    newFileList = elementList
    for i in range(lineListLen - 1):
        newFileList[i][0] = round((np.int64(elementList[i][0]) * (1e-7)), 7)
        newFileList[i][1] = round((np.int64(elementList[i][1]) * (1e-7)), 7)

    with open(file_dir_new + file_name_index + ".lab", 'w') as file_object:
        for i in range(lineListLen - 1):
            file_object.write(str(newFileList[i][0]) + ' ' + str(newFileList[i][1]) + ' ' + newFileList[i][2] + '\n')

shutil.rmtree("PJS_corpus_ver1.1_new/mono_label")
