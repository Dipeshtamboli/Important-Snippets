import sys
import os 
import glob
from os import walk

path_to_folder = "/home/dipesh/Desktop/age-gender/data/face-mask-data/Dataset/all_labeled_mask_office"
path2 = "/home/dipesh/Desktop/age-gender/office_data/"
f = []
# for i in walk(path2):
#   print (i)

# exit()
for (dirpath, dirnames, filenames) in sorted(walk(path_to_folder)):
# for (dirpath, dirnames, filenames) in walk(path2):
    # f.extend(filenames)
    # print(dirpath)
    # print(dirnames)
    for i in filenames:
        # print(dirpath)
        # print(dirnames)
        # print(i)
        # print(dirpath + '/' + i)
        f.append(dirpath + '/' + i)

    # for folders in dirnames:
    #   print(folders)
    # break
# print(f)
masked_list, non_masked_list = [], []
txt_mask = open("mask.txt", "a")

txt_no_mask = open("no_mask.txt", "a")
for f_n in sorted(f):
    # print(os.path.join(*f_n.split('/')[-2:]))
    # print(f_n.split('/')[-2][0]=='N')
    if (f_n.split('/')[-2][0]=='N'):
        non_masked_list.append(os.path.join(*f_n.split('/')[-2:]))
        txt_no_mask.write(os.path.join(*f_n.split('/')[-2:]) + "\n")
    if (f_n.split('/')[-2][0]=='M'):
        masked_list.append(os.path.join(*f_n.split('/')[-2:]))
        txt_mask.write(os.path.join(*f_n.split('/')[-2:])+ "\n")

txt_mask.close()
txt_no_mask.close()