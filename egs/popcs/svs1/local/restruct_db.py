from cgitb import text
from importlib.resources import path
import sys
import os
from praatio import textgrid

cnt1 = 0
cnt2 = 0
# str(cnt).zfill(4)

def writeTextgrid(arg1, arg2, file_name):
    # arg1: source file
    # arg2: target dir
    
    # target file
    arg2 = os.path.join(arg2, 'label')
    if os.path.exists(arg2) == False:
        os.makedirs(arg2)
    file_path = os.path.join(arg2, file_name + '.lab')

    # Try to open the file as textgrid
    tg = textgrid.openTextgrid(arg1, includeEmptyIntervals=False, duplicateNamesMode='rename')

    f = open(file_path, "w")  
    # TextGrid, List of Tiers, stored in tierNameList
    # Tier, List of Intervals, stored in entryList
    # Interval, List, including (start, end, lable)
 
    t = 0
    for nmTier in tg.tierNameList:
        wordTier = tg.tierDict[nmTier]
        for st, en, lab in wordTier.entryList:
            f.write("%f %f %s" % (t + st, t + en, lab) + '\n')
        t += wordTier.maxTimestamp - wordTier.minTimestamp
    f.flush()
    f.close()

def writeWav(arg1, arg2, file_name):
    # arg1: source file
    # arg2: target dir

    arg2 = os.path.join(arg2, 'wav')
    if os.path.exists(arg2) == False:
        os.makedirs(arg2)
    src_path = arg1
    trg_path = os.path.join(arg2, file_name + '.wav')
    os.system("cp " + src_path + ' ' + trg_path)
    

def walkFile(arg1, arg2):
    ### find '.TextGrid' and '.wav' in folder
    ### get raw_data 

    # arg1: soure_dir
    # arg2: target_dir
    file_list = os.listdir(arg1)
    for file_name in file_list:
        file_path = os.path.join(arg1, file_name)
        file_ext = file_path.rsplit('.', maxsplit=1)
        if file_ext[1] == 'TextGrid':
            global cnt1
            cnt1 = cnt1 + 1
            writeTextgrid(file_path, arg2, str(cnt1).zfill(4))
        elif file_ext[1] == 'wav':
            global cnt2
            cnt2 = cnt2 + 1
            writeWav(file_path, arg2, str(cnt2).zfill(4))

if __name__ == '__main__':
    arg1 = sys.argv[1] #source_dir
    arg2 = sys.argv[2] #target_dir
    for base_path, folder_list, file_list in os.walk(arg1):
        for folder_name in folder_list:
            dir_path = os.path.join(base_path, folder_name)
            walkFile(dir_path, arg2)
