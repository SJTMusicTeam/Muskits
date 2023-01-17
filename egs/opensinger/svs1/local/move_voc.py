import os
import shutil

if __name__ == '__main__':
    tgt_dir = '/data3/qt/else_data/opensinger'
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    src_dir = '/data3/qt/else_data/OpenSinger'
    for prefix_name in ['ManRaw', 'WomanRaw']:
        dir_path = os.path.join(src_dir, prefix_name)
        for dirname in os.listdir(dir_path):
            print(dirname)
            if dirname.endswith('Store'):
                continue
            singer_id, song_name = dirname.split('_')
            singer_name = prefix_name+'_'+singer_id
            tgt_singer_dir = os.path.join(tgt_dir, singer_name)
            if not os.path.exists(tgt_singer_dir):
                os.makedirs(tgt_singer_dir)
            
            for file in os.listdir( os.path.join(dir_path, dirname) ):
                if file.endswith('Store'):
                    continue
                src_file = os.path.join(dir_path, dirname, file)
                tgt_file = os.path.join(tgt_singer_dir, file)
                shutil.copy( src_file, tgt_file)
            print("Success!")
        #     break
        # break


