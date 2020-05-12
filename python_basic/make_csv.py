import os
import glob
from PIL import Image
import json


path_list = glob.glob("../extracted/*/*/*.jpg")

f = open("data.csv", "w")
f.write("train/test, class, path, width, height,MB_size \n")


train_class = {}
test_class = {}
data_mode = {"TEST":test_class,
             "MODEL":train_class}

for path in path_list:
    print(path)
    im = Image.open(path)
    mb_size = round(os.stat(path).st_size/1024/1024, 2)
    # print(mb_size)
    w,h = im.size
    class_num = path.split('/')[-2]
    train_or_test = path.split('/')[-3]
    # print(data_mode[train_or_test].keys())
    if class_num not in data_mode[train_or_test].keys():
        data_mode[train_or_test][class_num] = 1    
    else:
        data_mode[train_or_test][class_num] +=1
    # exit()
    # print(w,h)
    row = "{},{},{},{},{},{}\n".format(train_or_test,class_num,path,w,h,mb_size)
    f.write(row)
    # exit()

f.close()

# print(train_class)
# print(test_class)
print(data_mode)

with open('num_classes.json', 'w') as fp:
    json.dump(data_mode, fp)
