import glob
import os
import random
from xml.etree import ElementTree

import cv2
from sklearn.ensemble import RandomForestClassifier


###*Funkcja ładująca do słownika dane do testowania z inputu użytkownika*###


def order_load():
    i = 0
    order = input()
    if order == "classify":
        # print("Put your data here")
        n_files_s = input()  # number of files to analyze
        n_files = int(n_files_s)
        dicts = (
            []
        )  # list of dictionaries; each dictionary represents png file and its attributes
        while i < n_files:  # tyle iteracji ile plików
            # print(i)
            object = {}  # initializing dictionary; one for each png file
            file_1 = input()  # photo name
            object["file_name"] = file_1  # add file name to dictionary
            n_1 = input()  # number of frames to classify
            n_1i = int(n_1)
            object["to_classify"] = n_1i  # przypisanie ilosci wycinków do słownika
            j = 0
            while j < n_1i:  # ile kompletow wspolrzednych zebrac
                coordinates = input().split()
                int_coordinates = [int(c) for c in coordinates]
                xmin = int_coordinates[0]
                xmax = int_coordinates[1]
                ymin = int_coordinates[2]
                ymax = int_coordinates[3]
                bbox = (xmin, ymin, xmax, ymax)
                object[f"coordinates{j+1}"] = bbox
                j += 1
            dicts.append(object)
            i += 1
        # print(dicts)
        return dicts


###*Funkcja zwracająca listę przyciętych*###


def crop_photos(bbox_list, image_path):
    fragmented_photos = []
    img = cv2.imread(image_path)
    for b in bbox_list:
        xmin, ymin, xmax, ymax = b
        crop_img = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
        fragmented_photos.append(crop_img)

    return fragmented_photos


###*Funkcja zwracająca listę słowników {image: obrazek} wybranych i dociętych na podstawie komendy z terminala*###


def load_testdata_basedon_term(dicts, parent_path):

    fragmented_test_photos = []
    path_to_images = os.path.join(parent_path, "test", "images")
    for dic in dicts:
        file_name = dic["file_name"]
        path_to_image = os.path.join(path_to_images, file_name)
        m = 0
        bbox = []
        while m < dic["to_classify"]:
            bbox.append(dic[f"coordinates{m+1}"])
            m += 1
        fragmented_photo = crop_photos(bbox, path_to_image)
        for cp in fragmented_photo:
            sample = {"image": cp}
            fragmented_test_photos.append(sample)

    return fragmented_test_photos


###*Funkcja ładująca ścieżki xmli i png oraz wszystkie istotne informacje do listy słownikow, gdzie kazdy slownik to jeden obraz*###


def load_data_from_set(
    set,
    parent_path,
):
    path_to_annotations = os.path.join(parent_path, set, "annotations/*.xml") #path to annot
    path_to_images = os.path.join(parent_path, set, "images")  #path to images
    annotations_paths = glob.glob(path_to_annotations)
    #images_paths = glob.glob(path_to_images)


    set_data = []
    for ap in annotations_paths:
        single_file = {}
        dom = ElementTree.parse(ap)

        # przypisanie ścieżek do xml i do obrazków
        single_file["annotation_path"] = ap
        base_file_name = os.path.basename(ap).split('.')[0]
        single_file["image_path"] = os.path.join(path_to_images, f"{base_file_name}.png")

        tree = ElementTree.parse(ap)
        root = tree.getroot()
        fn = root.find("filename").text
        # print(fn)
        objects = root.findall(".//object")
        speedlimit_data = {"amount": 0, "bboxes": []}  # dict for speedlimitdata signs
        non_speedlimit_data = {"amount": 0, "bboxes": []}  # dict for other signs

        for object in objects:
            name = object.find("name").text
            xmin = object.find("bndbox/xmin").text
            ymin = object.find("bndbox/ymin").text
            xmax = object.find("bndbox/xmax").text
            ymax = object.find("bndbox/ymax").text
            bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
            if name == "speedlimit":
                speedlimit_data["bboxes"].append(bbox)
                speedlimit_data["amount"] += 1
            else:
                non_speedlimit_data["bboxes"].append(bbox)
                non_speedlimit_data["amount"] += 1

        single_file[
            "speedlimit_data"
        ] = speedlimit_data  # add dict for speedlimitdata sign to general dict under key
        single_file[
            "non_speedlimit_data"
        ] = non_speedlimit_data  # add dict for non_speedlimitdata sign to general dict under key

        set_data.append(single_file)  # add general dict to the list

    return set_data


def is_bbox_inblacklist(bbox, bboxes_blacklist):

    x_min, y_min, x_max, y_max = bbox
    for black_bbox in bboxes_blacklist:

        black_x_min, black_y_min, black_x_max, black_y_max = black_bbox
        if x_max < black_x_min or x_min > black_x_max:
            continue
        if y_max < black_y_min or y_min > black_y_max:
            continue
        return True

    return False


###*Funkcja zwracająca randomowe wycinki obrazów nie będące znakami w obrębie jednego zdjęcia*###


def get_random_crops(image_path, bboxes_blacklist, n):

    img = cv2.imread(image_path)
    dimensions = img.shape
    im_height = img.shape[0]
    im_width = img.shape[1]

    counter = 0
    picked_list = []
    while counter < 100:
        counter += 1
        x1 = random.randint(0, im_width)
        y1 = random.randint(0, im_height)
        r_width = random.randint(int(im_width / 5), int(im_width / 3))
        r_height = random.randint(int(im_height / 5), int(im_height / 3))
        bbox = (x1, y1, x1 + r_width, y1 + r_height)
        if (
            x1 + r_width >= im_width or y1 + r_height >= im_height
        ):  ###check if bbbox is in picture frame
            continue
        if is_bbox_inblacklist(
            bbox, bboxes_blacklist
        ):  ###check if bbox intersects with signs bboxes
            continue
        picked_list.append(bbox)
        if len(picked_list) == n:
            break

    return crop_photos(picked_list, image_path)


###*Funkcja przygotowująca (WYCINA FRAGMENTY, LABELUJE) dane uczące na podstawie listy slownikow*###


def prepare_samples(data, add_random=True):

    output = []

    for d in data:
        all_bboxes = []
        if d["speedlimit_data"]["amount"] != 0:
            speed_lim_boxes = d["speedlimit_data"]["bboxes"]
            speed_lim_images = crop_photos(speed_lim_boxes, d["image_path"])
            for sli in speed_lim_images:
                positive_sample = {"image": sli, "label": "speedlimit"}
                output.append(positive_sample)

        if d["non_speedlimit_data"]["amount"] != 0:
            non_speed_lim_boxes = d["non_speedlimit_data"]["bboxes"]
            non_speed_lim_images = crop_photos(non_speed_lim_boxes, d["image_path"])
            for sli in non_speed_lim_images:
                negative_sample = {"image": sli, "label": "non_speedlimit"}
                output.append(negative_sample)

        if add_random == True:
            all_bboxes = (
                d["speedlimit_data"]["bboxes"] + d["non_speedlimit_data"]["bboxes"]
            )
            random_crops = get_random_crops(d["image_path"], all_bboxes, 1)
            for crop in random_crops:
                # cv2.imshow("png",crop)
                # cv2.waitKey(0)
                negative_sample = {"image": crop, "label": "non_speedlimit"}
                output.append(negative_sample)

    return output


###*Funkcja dodająca featury do próbek na podstawie wyuczonego słownika*###


def add_features(data, vocabulary):

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    bow.setVocabulary(vocabulary)
    for sample in data:
        kpts = sift.detect(sample["image"], None)
        desc = bow.compute(sample["image"], kpts)
        if desc is None:  # check if desc is an array or is none(not defined)
            sample[
                "feature_vector"
            ] = None  # if there is not feature vector for a sample
        else:
            sample["feature_vector"] = desc[
                0
            ]  # desc[0] - to reduce to one dimensional array

    return data


###*Funkcja zwracająca ilosc znakow speedlimit i non_speedlimit na podstawie samples - listy slownikow{image:, label}*###


def sample_summary(samples):
    speed_limit_number = 0
    non_speed_limit_number = 0
    for s in samples:
        if s["label"] == "speedlimit":
            speed_limit_number += 1
        else:
            non_speed_limit_number += 1
    print(
        "Number of speedlimit samples: ",
        speed_limit_number,
        "\nNumber of NONspeedlimit samples: ",
        non_speed_limit_number,
    )


###*Uczy BoVW i zwraca słownik cech*###


def learn_bovw(data):

    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample["image"], None)
        kpts, desc = sift.compute(sample["image"], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    return vocabulary


###*trenuje model na podstawie featurów i labeli, zwraca nauczony model*###


def train(data):

    feature_vectors = []
    labels = []

    for sample in data:
        if sample["feature_vector"] is not None:
            feature_vectors.append(sample["feature_vector"])
            labels.append(sample["label"])

    rf = RandomForestClassifier(verbose=0)  # verbosity
    rf.fit(feature_vectors, labels)

    return rf


###*Dokonuje predykcji dla dostarczonych danych*###


def predict(model, single_data):

    if single_data["feature_vector"] is None:
        return "other"

    predicted_label = model.predict([single_data["feature_vector"]])[
        0
    ]  # 1 element array
    if predicted_label == "non_speedlimit":
        predicted_label = "other"

    # ------------------
    return predicted_label


def main():

    # BEGIN DIRECTORIES
    # current directory
    path = os.getcwd()
    # parent directory
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    # END DIRECTORIES

    ##BEGIN TRAINING MODEL (you can uncomment the prints if want to see whats goin' on :] )

    # lista słowników przechowujących istotne informacje o wszystkich zdjęciach z folderu TRAIN
    dane = load_data_from_set("train", parent_path)

    # przygotowanie danych z katalogu
    # print("Przygotowanie danych")
    train_samples = prepare_samples(dane)

    # sample_summary(train_samples)

    # uczenie bovwa zwrócenie słownika
    # print("Learn Bowv - get vobulary", flush=True)
    vocab = learn_bovw(train_samples)

    # dodanie ficzerów do danych treningowych
    # print("Adding feature-vectors to samples", flush=True)
    train_samples = add_features(train_samples, vocab)

    # trenowanie modelu
    # print("Start training rfc")
    model = train(train_samples)
    # print("END of training")

    ##END TRAINING MODEL

    predicted_labels_list = []
    target_labels_list = []

    ###BEGIN GET TEST DATA FROM CONSOLE

    zadane = order_load()
    demanded_croped_photos = load_testdata_basedon_term(zadane, parent_path)
    # samples based od on input order
    demanded_test_samples = add_features(demanded_croped_photos, vocab)

    ###END GET DATA FROM CONSOLE

    ###PREDYKCJA DLA DANYCH WCZYTANYCH Z TERMINALA

    for tr in demanded_test_samples:
        predicted_label = predict(model, tr)
        print(predicted_label)


if __name__ == "__main__":
    main()
