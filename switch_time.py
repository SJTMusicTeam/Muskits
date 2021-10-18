import numpy as np
for file_index in range(100):
    if(file_index<9):
        file_name_index = "0" + str(file_index + 1)
    else:
        file_name_index = str(file_index + 1)
    if file_index<100:
        with open("pjs0"+file_name_index+".lab", 'r') as f:
            file = f.read()
    else:
        with open("pjs" + file_name_index + ".lab", 'r') as f:
            file = f.read()

    lineList = file.split('\n')

    lineListLen = len(lineList)
    elementList = []
    for i in range(lineListLen-1):
        elementList.append(lineList[i].split(' '))
    '''
    def str2num(str):
        strLen = len(str)
        num = 0
        for i in range(strLen):
            num += np.int64(str[strLen-i-1])*(pow(10,i))
        return num
    '''
    newFileList = elementList
    for i in range(lineListLen-1):
        newFileList[i][0] = round((np.int64(elementList[i][0]) * (1e-7)) , 7)
        newFileList[i][1] = round((np.int64(elementList[i][1]) * (1e-7)) , 7)
        print(newFileList[i])

    with open(file_name_index+".lab", 'w') as file_object:
        for i in range(lineListLen-1):
            file_object.write(str(newFileList[i][0])+' '+str(newFileList[i][1])+' '+newFileList[i][2]+'\n')