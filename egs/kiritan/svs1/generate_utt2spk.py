import os
import numpy as np

path_root = 'data/'
path_leaf = ['train/', 'eval1/', 'dev/']
for path_i in path_leaf:
	f_1 = open(path_root + path_i + 'utt2spk', 'w')
	f_1.close()
	f_2 = open(path_root + path_i + 'spk2utt', 'w')
	f_2.close()
	with open(path_root + path_i + 'segments') as f:
	    for line in f:
	    	th = line.split(' ')
	    	uttid_i = th[0]
	    	with open(path_root + path_i + 'utt2spk', 'a') as f_1:
	    		f_1.write(str(uttid_i) + '\t' + 'kiritan' + "\n")
	    	with open(path_root + path_i + 'spk2utt', 'a') as f_2:
	    		f_2.write('kiritan' + '\t' + str(uttid_i) + "\n")

