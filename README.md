### road_sign_detection_ai_lab
 Project realised as a final task in *Introduction to Artificial Intelligence* classes on Automatic control and robotics faculty, at Poznan University of Technology.
 
 ## Table of contents
* [Informacje ogólne](#informacje-ogólne)
* [Technologie](#technologie)
* [Opis działania programu](#opis-działania)
* [Użyte biblioteki](#setup)

## Informacje ogólne
Celem projektu jest konstrukcja rozwiązania opartego na Machine Learning, zdolnego do klasyfikacji znaków drogowych udokumentowanych na zdjęciach. Program ma za zadanie rozpoznać czy
na narzuconym wycinku zdjęcia znajduje się znak ograniczenia prędkości - klasa "speedlimit", czy też inny (lub na podanym wycinku nie ma znaku) - klasa "other".

## Technologie
Podczas realizacji projektu korzystano z **Python 3.9**, natomiast środowiskiem programistycznym był **PyCharm Community Edition 2021.2.2.**
	
## Opis działania
Po uruchomieniu programu zapisanego w pliku *main.py* następuje wykonanie zapisanych w funkcji *main* poleceń. W pierwszej kolejności w sekcji *#DIRECTORIES* przechwytywane i
zapisane do zmiennych są ścieżki: ścieżka do pliku programu, oraz ścieżka do folderu nadrzędnego, w którym znajdują się foldery *test* i *train*.
 
W celu wytrenowania modelu potrzebne są dane treningowe. Do ich wczytania używana jest funkcja *load_data_from_set(set, parent_path)*. Argument funkcji - *set* - służy do określenia 
folderu, z którego wczytane mają zostać dane, natomiast *parent_path* to ściezka folderu nadrzędnego wobec pliku porgramowego .py. Funkcja ta ma za zadanie parsować przyporządkowane
poszczególnym zdjęciom pliki .xml, tak by skojarzyć z odpowiednimi zdjęciami poszczególne wyłuskane dane. Funkcja zwraca listę słowników, w którym każdy słownik reprezentuje
jedno zdjęcie ze zbioru. Słownik zawiera ścieżkę do pliku .png, .xml, a także informację o występowaniu i liczebności znaków z danej kategorii
("speedlimit" oraz "non_speedlimit") oraz współrzędne ramek ich położenia na obrazie. 

Tak przygotowany zbiór trafia następnie do funkcji *prepare_samples(data, add_random = True)*, która jako argumenty przyjmuje wyżej opisaną listę oraz argument logiczny *add_random*,
domyślnie *True* - który decyduje o tym, czy do zbioru treningowego dodawane będą losowe fragmenty zdjęć, nie zawierające elementów zdefiniowanych w plikach .xml znaków drogowych.
Na podstawie informacji czy dany znak należy do kategorii "speedlimit" czy też "nonspeedlimit" funkcja wycina odpowiednie fragmenty zdjęcia zawierające znak oraz nadaje im odpowiedni
label. Do wycinania ramek wykorzystywana jest funkcja *crop_photos(bbox_list, image_path)*, która jako argumenty przyjmuje listę współrzędnych ramek oraz ścieżkę do zdjęcia z obrębu
którego nastąpi wycinanie. 
W przypadku gdy zmienna *add_random = True* obsłużone zostaje również wczytanie losowych wycinków ubogacających zbió treningowy. W tym celu wykorzystywana
jest funkcja *get_random_crops(image_path, bboxes_blacklist, n)* gdzie *image_path* do ścieżka pojedynczego zdjęcia, *bboxes_blacklist* to lista współrzędnych ramek zawierających
znaki drogowe w obrębie danego zdjęcia, a *n* to liczba "randomowych" wycinków, które mają zostać pozyskane z danego zdjęcia. Wewnątrz tej metody wykorzystywana jest funkcja 
*is_bbox_inblacklist(bbox, bboxes_blacklist)*, która sprawdza czy ramka o wylosowanych współrzędnych nie przecina się z ramkami zawierającymi znaki drogowe. Końcowo funkcja *prepare_samples* zwraca słowniki, gdzie każdy z nich zawiera wycięty fragment zdjęcia oraz odpowiedni label "speedlimit" - znaki ograniczenia prędkości, oraz "non_speedlimit" -
inne znaki losowe fragmenty (elementy krajobrazu, krzewy, fragmenty tła etc.)

Tak przygotowane dane trafiają do funkcji *learn_bovw(data)*, która korzystając z dostarczanych przez bibliotekę opencv metod (*BOWKMeansTrainer* oraz *SIFT*) zwraca słownik cech
wyróżnionych na podstawie dostarczonych obrazów.

Następnie przygotowane wcześniej próbki poddane są działaniu funkcji *add_features(data, vocabulary)*, która jako argumenty przyjmuje przygotowane wcześniej dane oraz utworzony
słownik. Funkcja na podstawie analizy dostarczonych obrazów, przyporządkowuje każdemu odpowiedni wektor cech. 

Uzyskane komplety danych trafiają do funkcji *train(data)*. Efektem jej działania jest zwrócenie nauczonego modelu, który korzysta z dostarczanego przez bibliotekę *sklearn.enseble*
estymatora *RandomForestClassifier*. Model trenowany jest na podstawie deskryptorów przyporządkowanych danemu obrazowi jak i labelów informujących czy na danym wycinku znajduje się
znak "speedlimit". W momencie zakończenia wszystkich powyższych etapów w konsoli wyświetla się komunikat *"END of training"*.

Następnym etapem działania programu jest wczytanie danych użytkownika w standardzie zgodnym z przedstawionym w instrukcji do projektu. Funkcjonalność ta zostaje zrealizowana przez 
funkcję *order_load()*, która zwraca listę słowników, w której każdy słownik reprezentuje pojedyncze zdjęcie ze zbioru testowego oraz dane o ramkach, które mają zostać z niego 
ekstrahowane i poddane predykcji. By całość zadziałała prawidłowo struktura danych wpisywanych powinna wyglądać zgodnie z informacjami zawartymi w instrukcji (przykładowy input):
- *classify*
- *1* - liczba zdjęć z których analizowane będą wycinki
- *road231.png* - nazwa pliku
- *2* liczba wycinków do analizy
- *82 52 209 218* - współrzędne 1. wycinka oddzielone spacjami
- *127 191 179 214* - współrzędne 2. wycinka odzielone spacjami

Tak wczytane dane poddane są działaniu funkcji *load_data_basedon_term(dicts, parent_path)*, która na podstawie nazw plików wprowadzonych w
konsoli oraz współrzędnych ramek, zwraca listę wyciętych obrazów testowych. Następnie tak przygotowane dane poddane są działaniu funkcji *add_features*, której działanie zostało 
opisane powyżej. Następnie w pętli iterującej po każdym elemencie w liście testowych fragmentów, program dokonuje predykcji realizowanej przez funkcję *predict(model, single_data)*,
która jako argument przyjmuje wytrenowany model, a także dany fragment testowy. Funkcja zwraca łańcuch znaków *predicted_label*, który następnie printowany jest w konsoli w postaci:
**"speedlimit"** - jeśli wykryty został znak ograniczenia prędkości, lub **"other"** jeśli inny (ew. fragment otoczenia nie będący znakiem). 


## Setup
W celu uruchomienia programu należy zainstalować/zaimportować poniższe biblioteki:

```
$ import glob
$ import os
$ import random
$ from xml.etree import ElementTree
$ import cv2
$ from sklearn.ensemble import RandomForestClassifier
```
 
