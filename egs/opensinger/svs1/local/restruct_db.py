
from pathlib import Path
import sys
import os


cnt = 0
# str(cnt).zfill(4)


def writeWav(arg1, arg2, file_name):
    # arg1: source file
    # arg2: target dir

    arg2 = os.path.join(arg2)
    if os.path.exists(arg2) == False:
        os.makedirs(arg2)
    src_path = arg1
    trg_path = os.path.join(arg2, file_name)
    if not os.path.exists(trg_path):
        os.system("cp " + src_path + ' ' + trg_path)
    

def walkFile(arg1, arg2):
    ### find '.TextGrid' and '.wav' in folder
    ### get raw_data 

    # arg1: soure_dir
    # arg2: target_dir
    file_list = [ str(p) for p in Path(arg1).rglob('*.wav') ]
    file_list2 = [ str(p) for p in Path(arg2).rglob('*.wav') ]
    if len(file_list) == len(file_list2):
        return
    for file_name in file_list:
        items = file_name.split('/')
        gender = items[-3]
        singer_id = items[-2].split("_")[0].zfill(4)
        segment_id = items[-1].split("_")[1:]
        segment_id = segment_id[0]+segment_id[1][:-4].zfill(4)+'.wav'

        writeWav(file_name, arg2, gender+singer_id+'_'+segment_id)

if __name__ == '__main__':
    arg1 = sys.argv[1] #source_dir
    arg2 = sys.argv[2] #target_dir
    for base_path, folder_list, file_list in os.walk(arg1):
        for folder_name in folder_list:
            dir_path = os.path.join(base_path, folder_name)
            walkFile(dir_path, arg2)
            
