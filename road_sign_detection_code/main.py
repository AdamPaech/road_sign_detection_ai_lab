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

def order_load():
    i = 0
    order = input()
    if order == "classify":
        print("jest")
        n_files_s = input() #ile jest plikow
        n_files = int(n_files_s)
        #print(n_files)
        object = {}
        dicts = []
        while(i<n_files):   #tyle iteracji ile plików
            # print(i)
            file_1 = input() #nazwa zdjecia
            object['file_name'] = file_1 #przypisanie nazwy do słownika
            n_1 = input()
            n_1i = int(n_1)
            object['to_classify'] = n_1i #przypisanie ilosci wycinków do słownika
            j=0
            while(j<n_1i):  #ile kompletow wspolrzednych zebrac
                coordinates = input().split()
                object['coordinates'] = coordinates
                j+=1
            ###po kazdym pliku wpierolic: nazwe,
            dicts.append(object)
            i+=1
        print(dicts)








order_load()










