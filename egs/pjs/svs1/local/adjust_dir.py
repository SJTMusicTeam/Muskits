import os
import shutil

'''
os.rmdir("PJS_corpus_ver1.1_new/wav")
os.rmdir("PJS_corpus_ver1.1_new/midi_label")
os.rmdir("PJS_corpus_ver1.1_new/mono_label")
os.rmdir("PJS_corpus_ver1.1_new")
'''
os.mkdir("PJS_corpus_ver1.1_new")
os.mkdir("PJS_corpus_ver1.1_new/wav")
os.mkdir("PJS_corpus_ver1.1_new/midi_label")
os.mkdir("PJS_corpus_ver1.1_new/mono_label")
file_list = os.listdir("PJS_corpus_ver1.1")
print(file_list)

for i in file_list:
    if i.find("pjs") != -1 and not (i.find("pdf") != -1):
        shutil.copy("PJS_corpus_ver1.1/" + i + "/" + i + "_song.wav",
                    "PJS_corpus_ver1.1_new/wav")
        shutil.copy("PJS_corpus_ver1.1/" + i + "/" + i + ".mid",
                    "PJS_corpus_ver1.1_new/midi_label")
        shutil.copy("PJS_corpus_ver1.1/" + i + "/" + i + ".lab",
                    "PJS_corpus_ver1.1_new/mono_label")
