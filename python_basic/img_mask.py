import time
import shutil
import os
import cv2
from xml_loading import get_annotations
import numpy as np
import glob

start_time = time.time()

# def quick_load(annotation_path):
#     complete_list = get_annotations(annotation_path)
#     for i in range(len(complete_list)):
#         img = cv2.imread(annotation_path[:-3]+"jpg")
#         try:
#             print(img.shape)
#         except AttributeError as e:
#             print("AttributeError occured")
#             print("skipped this file: {}".format(annotation_path))
#             continue
#         # img = np.zeros( (image.shape[0],image.shape[1]) ) # create a single channel (of image size) pixel black image 

#         for j in range(len(complete_list[i])):
#             print(i,j)
#             cont = np.array([[int(vertex.attrib['X']),int(vertex.attrib['Y'])] for vertex in complete_list[i][j]])
#             print(cont.shape)
#             cv2.fillPoly(img, pts =[cont], color=(255,255,255))
#         cv2.imwrite("cont_annot{}.jpg".format(i), img) 
#     exit()

# quick_load("9493-121-2.xml")
path_list = glob.glob("../new_extracted/MODEL/*/*.xml")

def create_dir(path_to_img):
    directory = os.path.join(*path_to_img.split('/')[:-1]) 
    if not os.path.exists(directory):
        os.makedirs(directory)    

def save_img(save_path, img):
    create_dir(save_path)
    cv2.imwrite(save_path, img)

for annotation_path in path_list:
    jpg_path = annotation_path[:-3]+"jpg"
    image = cv2.imread(jpg_path)
    # print(image.shape)
    try:
        img_shape = (image.shape)
        # print(image.shape)
    except AttributeError as e:
        print("jpg for file: {}".format(annotation_path))

        try:
            jpg_path = annotation_path[:-3]+"JPG"
            image = cv2.imread(jpg_path)
            img_shape = (image.shape)
            print("but JPG was available")
        except AttributeError as e:

            print("AttributeError occured")
            print("skipped this file: {}".format(annotation_path))
            dest = shutil.copy(annotation_path, "./xml_without_jpg")  
            continue

    # print(annotation_path)
    class_number = annotation_path.split('/')[-2]
    img_name = annotation_path.split('/')[-1][:-4]

    save_orig_path = "../id_wise_annotated_imgs/colour_all/" + class_number + '/' + img_name+".jpg"

    save_img(save_orig_path, image)
    complete_list, annotation_id, LineColor_list = get_annotations(annotation_path)

    assert (len(complete_list) == len(annotation_id))
    for i in range(len(complete_list)):
        img = cv2.imread(jpg_path)
        bin_img = np.zeros( (image.shape[0],image.shape[1]) ) # create a single channel (of image size) pixel black image 
        for j in range(len(complete_list[i])):
            # print(i,j)
            cont = np.array([[int(vertex.attrib['X']),int(vertex.attrib['Y'])] for vertex in complete_list[i][j]])
            # print(cont.shape)
            cv2.fillPoly(img, pts =[cont], color=(255,255,255))
            cv2.fillPoly(bin_img, pts =[cont], color=(255,255,255))
        save_colour_path = "../id_wise_annotated_imgs/colour_all/" + class_number + '/' + img_name+"_{}.jpg".format(annotation_id[i])
        save_binary_path = "../id_wise_annotated_imgs/binary_masks/" + class_number + '/' + img_name+"_{}.jpg".format(annotation_id[i])
        save_linecolour_path = "../id_wise_annotated_imgs/annot_id_{}/".format(annotation_id[i]) + class_number + '/' + img_name+"_{}.jpg".format(annotation_id[i])
        save_img(save_colour_path, img)
        save_img(save_linecolour_path, img)
        save_img(save_binary_path, bin_img)


end = time.time()
print('time taken is: {} hrs, {} min'.format((end-start_time)//3600,(end-start_time)//60%60 ))