import os
import glob
from xml.etree import ElementTree
import shutil

# ####path stuff
#
# #current directory
# path = os.getcwd()
# print("Current Directory", path)
#
# #parent directory
# parent_path = os.path.abspath(os.path.join(path, os.pardir))
# print("Parent Directory", parent_path)
#
# #dane -> annotations path
# path_to_xml = os.path.join(parent_path, 'dane/annotations/*.xml')
# print("Path to xml", path_to_xml)
# ###test
# # for file in glob.glob(path_to_xml):
# #     dom = ElementTree.parse(file)
# #     name = dom.findall('object/name')
# #
# #     for n in name:
# #             print(n.text)
#
#
# for file in glob.glob(path_to_xml): #iteracja po wszystkich xmlach w katalogu
#
#     # full_file = os.path.abspath(os.path.join('dane', 'annotations', file))
#
#     dom = ElementTree.parse(file)
#     png_name = dom.findall('filename')
#     for p in png_name:
#         ori_png = os.path.join(parent_path,'dane', 'images', p.text)   ####
#
#     name = dom.findall('object/name')
#     found = 0;
#     for n in name:
#         if n.text == "speedlimit":
#             found = 1;
#
#     if found == 1:
#         shutil.copy2(file, os.path.join(parent_path, 'speedlimit/annotations')) #moving xml file speedlimit cat.
#         shutil.copy2(ori_png, os.path.join(parent_path, 'speedlimit/images')) #moving assigned png image to speedlimit cat.
#         print('moved:', ori_png)
#     else:
#         shutil.copy2(file, os.path.join(parent_path, 'no_speedlimit/annotations'))  # moving xml file
#         shutil.copy2(ori_png, os.path.join(parent_path, 'no_speedlimit/images'))  # moving assigned png file
#
# # i = 1
# # for f in glob.glob(os.path.join(parent_path, 'speedlimit/annotations/*.xml')):
# #     png_name1 = dom.findall('filename')
# #     for p in png_name1:
# #         ori_png1 = os.path.join(parent_path, 'dane', 'images', p.text)  ####
# #     if i%4 == 0:
# #         shutil.copy2(f, os.path.join(parent_path, 'test/annotations'))
# #         shutil.copy2(ori_png1, os.path.join(parent_path, 'test/images'))
# #
# #     else:
# #         shutil.copy2(f, os.path.join(parent_path, 'train/annotations'))
# #         shutil.copy2(ori_png1, os.path.join(parent_path, 'train/images'))
# #     i+=1
#
# # for f in glob.glob(parent_path, 'speedlimit/annotations/*.xml'):

#funkcja do ładowania danych z input do fazy testowania
import cv2


def order_load():
    i = 0
    order = input()
    if order == "classify":
        print("jest")
        n_files_s = input()             #number of files to analyze
        n_files = int(n_files_s)
        dicts = []                      #list of dictionaries; each dictionary represents png file and its attributes
        while(i<n_files):               #tyle iteracji ile plików
            # print(i)
            object = {}                  #initializing dictionary; one for each png file
            file_1 = input()             #photo name
            object['file_name'] = file_1 #add file name to dictionary
            n_1 = input()                #number of frames to classify
            n_1i = int(n_1)
            object['to_classify'] = n_1i #przypisanie ilosci wycinków do słownika
            j=0
            while(j<n_1i):               #ile kompletow wspolrzednych zebrac
                coordinates = input().split()
                object[f'coordinates{j+1}'] = coordinates
                j+=1
            ###po kazdym pliku wpierolic do listy
            dicts.append(object)
            i+=1
        print(dicts)
        return dicts

# order_load()

# #current directory
path = os.getcwd()
#parent directory
parent_path = os.path.abspath(os.path.join(path, os.pardir))

def load_data_from_set(set, parent_path):
    path_to_annotations = os.path.join(parent_path, set, 'annotations/*.xml')
    path_to_images = os.path.join(parent_path, set, 'images/*.png')
    annotations_paths = glob.glob(path_to_annotations)
    images_paths = glob.glob(path_to_images)

    i = 0
    set_data = []
    for ap in annotations_paths:
        single_file = {}
        found = 0
        dom = ElementTree.parse(ap)
        names = dom.findall('object/name')
        # print(names)
        counter = 0
        for n in names:
            # print(n.text)
            if n.text == "speedlimit":
                found = 1;
                counter+=1;
        if(found==1):
            j = 0
            # set_data.append(
            #     {'annotation_path': annotations_paths[i], 'image_path': images_paths[i], 'is_speedlimit': 1,
            #          'amount': counter})

            single_file['annotation_path'] = annotations_paths[i]
            single_file['image_path'] = images_paths[i]
            single_file['is_speedlimit'] = 1
            single_file['amount'] = counter

        else:
            single_file['annotation_path'] = annotations_paths[i]
            single_file['image_path'] = images_paths[i]
            single_file['is_speedlimit'] = 0
            single_file['amount'] = counter

        ####teraz dostać sie do xów i ygów
        tree = ElementTree.parse(ap)
        root = tree.getroot()
        fn = root.find('filename').text
        print(fn)
        objects = root.findall('.//object')
        k = 1
        for object in objects:
            name = object.find('name').text
            if name == "speedlimit":
                xmin = object.find('bndbox/xmin').text
                ymin = object.find('bndbox/ymin').text
                xmax = object.find('bndbox/xmax').text
                ymax = object.find('bndbox/ymax').text
                single_file[f'xmin{k}'] = xmin
                single_file[f'ymin{k}'] = ymin
                single_file[f'xmax{k}'] = xmax
                single_file[f'ymax{k}'] = ymax
                k += 1

            # print(name, xmin, ymin, xmax, ymax)

        set_data.append(single_file)
        i+=1

    return set_data


list = load_data_from_set('train', parent_path)
###wyswietlanie listy
# for l in list:
#     print(l)

###wycinanie próba###
path1 = r'D:\\AIR\\Semestr 5\\AI_PROJEKT\\train\\images\\road544.png'
img = cv2.imread(path1)
crop_img = img[223:260,107:145]
cv2.imshow("png", crop_img)
cv2.waitKey(0)












