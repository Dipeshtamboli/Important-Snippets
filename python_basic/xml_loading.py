import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd
import pdb

def  get_annotations(annotation_path):
    tree = ET.parse(annotation_path)
    # complete_list = [[],[],[]]
    # complete_list = [[],[]]
    complete_list = []
    annotation_id = []
    LineColor_list = []
    l_i =-1
    for Annotation in tree.getroot().findall('Annotation'):
        # print(Annotation)    
        # pdb.set_trace()
        l_i +=1
        complete_list.append([])
        annotation_id.append(int(Annotation.attrib['Id']))
        LineColor_list.append(int(Annotation.attrib['LineColor']))

        # print("l_i",l_i)
        for Regions in Annotation.findall('Regions'):
            # print(Regions)
            for Region in Regions.findall('Region'):
                # print(Region)
                # Region_list.append(Region)
                # complete_list[l_i].append(Region)
                for Vertices in Region.findall('Vertices'):
                    # print(Vertices)            
                    # print(len(Vertices))
                    # if l_i >1:
                    #     break
                    complete_list[l_i].append(Vertices)
                    # point_list.append(Vertices)
                    # for Vertex in Vertices.findall('Vertex'):
                    #     continue
                        # print(Vertex.attrib['X'], Vertex.attrib['Y'])            
                        # break
    return complete_list, annotation_id, LineColor_list



# print(len(complete_list))
# print(len(complete_list[0]),len(complete_list[0][0]),len(complete_list[0][1]),len(complete_list[0][2]))
# print(len(complete_list[1]))
# # print(len(point_list))
# # print(len(Region_list))
# pdb.set_trace()
# exit()

# All_points =[]
# for Vertex in tree.getroot().findall('Annotation/Regions/Region/Vertices/Vertex'):
#     # print(Vertex)
#     # print(Vertex.attrib['X'], Vertex.attrib['Y'])
#     All_points.append((Vertex.attrib['X'], Vertex.attrib['Y']))

# print(len(All_points))


if __name__ == '__main__':
    # annotation_path = "/home/dipesh/skinai/codes/annot_3/382-102-1.xml"
    annotation_path = "/home/dipesh/skinai/new_extracted/MODEL/10/790-10-1.xml"

    complete_list, annotation_id, LineColor_list = get_annotations(annotation_path)

    # pdb.set_trace()

    # print((complete_list[0][0][1][0].attrib['X'], complete_list[0][0][1][0].attrib['Y']))
    # print("first point: ",(complete_list[0][0][0].attrib['X'], complete_list[0][0][0].attrib['Y']))
    # complete_list[0][0][0].attrib['X']
    for i in complete_list:
        print("regions:{}".format(len(i)))
        for j in i:
            print("--{}".format(len(j)))

    print(annotation_id)
    print(LineColor_list)

    pdb.set_trace() 