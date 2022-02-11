import os
import glob
from xml.etree import ElementTree
import shutil
import cv2
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm


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


###*funkcja obsługująca wejście do testowania*###

###*Funkcja ładująca do słownika dane do testowania z inputu użytkownika*###

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
                int_coordinates = [int(c) for c in coordinates]
                xmin = int_coordinates[0]
                xmax = int_coordinates[1]
                ymin = int_coordinates[2]
                ymax = int_coordinates[3]
                bbox = (xmin, ymin, xmax, ymax)
                object[f'coordinates{j+1}'] = bbox
                j+=1
            dicts.append(object)
            i+=1
        # print(dicts)
        return dicts

###*Funkcja zwracająca listę przyciętych znaków speedlimit*###

def crop_photos(bbox_list, image_path):
    fragmented_photos = []
    img = cv2.imread(image_path)
    for b in bbox_list:
        xmin, ymin, xmax, ymax = b
        crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        fragmented_photos.append(crop_img)

    return fragmented_photos

def load_testdata_basedon_term(dicts, parent_path):

    fragmented_test_photos = []
    path_to_images = os.path.join(parent_path, 'test', 'images')
    for dic in dicts:
        file_name = dic['file_name']
        path_to_image = os.path.join(path_to_images, file_name)
        m = 0
        bbox = []
        while(m < dic['to_classify']):
            bbox.append(dic[f'coordinates{m+1}'])
            m+=1
        fragmented_photo = crop_photos(bbox, path_to_image)
        for cp in fragmented_photo:
            sample = {'image': cp}
            fragmented_test_photos.append(sample)

    return fragmented_test_photos



###*Funkcja ładująca ścieżki xmli i png oraz wszystkie istotne informacje do listy słownikow, gdzie kazdy slownik to jeden obraz*###

def load_data_from_set(set, parent_path,):
    path_to_annotations = os.path.join(parent_path, set, 'annotations/*.xml')
    path_to_images = os.path.join(parent_path, set, 'images/*.png')
    annotations_paths = glob.glob(path_to_annotations)
    images_paths = glob.glob(path_to_images)

    i = 0
    set_data = []
    for ap in annotations_paths:
        single_file = {}
        dom = ElementTree.parse(ap)

        #przypisanie ścieżek do xml i do obrazków
        single_file['annotation_path'] = annotations_paths[i]
        single_file['image_path'] = images_paths[i]


        tree = ElementTree.parse(ap)
        root = tree.getroot()
        fn = root.find('filename').text
        # print(fn)
        objects = root.findall('.//object')
        speedlimit_data = {'amount': 0, 'bboxes':[]}            #dict for speedlimitdata signs
        non_speedlimit_data = {'amount': 0, 'bboxes': []}       #dict for other signs

        for object in objects:
            name = object.find('name').text
            xmin = object.find('bndbox/xmin').text
            ymin = object.find('bndbox/ymin').text
            xmax = object.find('bndbox/xmax').text
            ymax = object.find('bndbox/ymax').text
            bbox = (xmin, ymin, xmax, ymax)
            if(name == "speedlimit"):
                speedlimit_data['bboxes'].append(bbox)
                speedlimit_data['amount'] +=1
            else:
                non_speedlimit_data['bboxes'].append(bbox)
                non_speedlimit_data['amount'] += 1

        single_file['speedlimit_data'] = speedlimit_data             #add dict for speedlimitdata sign to general dict under key
        single_file['non_speedlimit_data'] = non_speedlimit_data     #add dict for non_speedlimitdata sign to general dict under key

        set_data.append(single_file)                                 #add general dict to the list
        i+=1

    return set_data


###*Funkcja przygotowująca (WYCINA FRAGMENTY) dane uczące na podstawie listy slownikow*###

def prepare_samples(data):

    output = []
    for d in tqdm(data):

        if d['speedlimit_data']['amount'] != 0:
            speed_lim_boxes = d['speedlimit_data']['bboxes']
            speed_lim_images = crop_photos(speed_lim_boxes, d['image_path'])
            for sli in speed_lim_images:
                positive_sample = {'image': sli, 'label':"speedlimit"}
                output.append(positive_sample)

        if d['non_speedlimit_data']['amount'] != 0:
            non_speed_lim_boxes = d['non_speedlimit_data']['bboxes']
            non_speed_lim_images = crop_photos(non_speed_lim_boxes, d['image_path'])
            for sli in non_speed_lim_images:
                negative_sample = {'image': sli, 'label':"non_speedlimit"}
                output.append(negative_sample)

    return output

###*Funkcja dodająca featery

def add_features(data, vocabulary):

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    bow.setVocabulary(vocabulary)
    for sample in tqdm(data):
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        if desc is None:     #check if desc is an array or is none(not defined)
            sample['feature_vector'] = None  #if there is not feature vector for a sample
        else:
            sample['feature_vector'] = desc[0] # desc[0] - to reduce to one dimensional array

    return data

###*Funkcja zwracająca ilosc znakow speedlimit i non_speedlimit na podstawie listy slownikow{image:, label}*###

def sample_summary(samples):
    speed_limit_number = 0
    non_speed_limit_number = 0
    for s in samples:
        if s['label'] == "speedlimit":
            speed_limit_number +=1
        else:
            non_speed_limit_number +=1
    print("Number of speedlimit samples: ", speed_limit_number,
          "\nNumber of NONspeedlimit samples: ", non_speed_limit_number)


###*Uczy BoVW i zwraca słownik*###

def learn_bovw(data):
    """
    Learns BoVW dictionary and saves it as "voc.npy" file.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    @return: Nothing
    """
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in tqdm(data):
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    return vocabulary

###*trenuje model na podstawie wektora featurów i labeli, zwraca nauczony model*###

def train(data):
    """
    Trains Random Forest classifier.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Trained model.
    """


    feature_vectors = []
    labels = []

    for sample in data:
        if sample['feature_vector'] is not None:
            feature_vectors.append(sample['feature_vector'])
            labels.append(sample['label'])

    rf = RandomForestClassifier(verbose=0) #verbosity
    rf.fit(feature_vectors, labels)

    return rf

###*Dokonuje predykcji dla dostarczonych
def predict(rf, data):
    """
    Predicts labels given a model and saves them as "label_pred" (int) entry for each sample.
    @param rf: Trained model.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Data with added predicted labels for each sample.
    """
    # perform prediction using trained model and add results as "label_pred" (int) entry in sample

    # for sample in data:
    # sample.update({'prediction':rf.predict(sample['desc'])[0]})
    for sample in data:
        if sample['desc'] is not None:
            predict = rf.predict(sample['desc'])
            sample['label_pred'] = int(predict)
    # ------------------
    return data


def main():


    #BEGIN DIRECTORIES
    # current directory
    path = os.getcwd()
    # parent directory
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    #END DIRECTORIES

    # lista słowników przechowujących istotne informacje o wszystkich zdjęciach z folderu TRAIN
    dane = load_data_from_set('train', parent_path)

    # pathpom = dane[0]['image_path']
    # bnbSL = dane[0]['speedlimit_data']['bboxes']
    # bnbNSL = dane[0]['non_speedlimit_data']['bboxes']



    # przyciete = crop_photos(bnbNSL, pathpom)
    # print("bnbSL", bnbSL, "N", bnbNSL)
    # for pr in przyciete:
    #     cv2.imshow("png",pr)
    #     cv2.waitKey(0)

    ##BEGIN TRAINING MODEL

    print("Przygotowanie danych")
    train_samples = prepare_samples(dane)

    sample_summary(train_samples)

    print("Learn Bowv - get vobulary", flush=True)
    vocab = learn_bovw(train_samples)

    print("Adding feature-vectors to samples", flush=True)
    train_samples = add_features(train_samples, vocab)

    print("Start training rfc")
    model = train(train_samples)
    print("END training rfc")

    ##END TRAINING MODEL

    ###BRUDNOPIS
    print("Wyswietlam fotki: ")
    zadane = order_load()
    # for z in zadane:
    #     print(z)
    demanded_croped_photos = load_testdata_basedon_term(zadane, parent_path)
    print("Adding feature-vectors to TEST samples", flush=True)
    demanded_test_data = add_features(demanded_croped_photos, vocab)                 ##adding features to samples

    # for zd in demanded_croped_photos:
    #     cv2.imshow("png",zd['image'])
    #     cv2.waitKey(0)

    ###END BRUDNOPIS

    # #BEGIN TEST
    #
    # #####test####
    # # lista danych z folderu TEST
    # dane_test = load_data_from_set('test', parent_path)
    # test_samples = prepare_samples(dane_test)
    #
    # print("Adding feature-vectors to TEST samples", flush=True)
    # test_samples = add_features(test_samples, vocab)
    #
    #
    # predicted_labels_list = []
    # target_labels_list = []
    # for tr in tqdm(test_samples):
    #      # cv2.imshow("png",tr['image'])
    #      # print("Label: ", tr['label'])
    #      try:
    #         predicted_label = model.predict([tr['feature_vector']]) #1 element array
    #      except ValueError as e:
    #         continue
    #         print(e)
    #      predicted_labels_list.append(predicted_label)
    #      target_labels_list.append(tr['label'])
    #
    # print("Summary: ")
    # conf = confusion_matrix(target_labels_list, predicted_labels_list, labels=["speedlimit", "non_speedlimit"])
    # print(conf)
    # accur = accuracy_score(target_labels_list, predicted_labels_list)
    # print(accur)


         # print("predicted label: ", predicted_label)
         # cv2.waitKey(0)



if __name__ == "__main__":
    main()













