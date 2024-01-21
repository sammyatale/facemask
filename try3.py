import numpy as np
import cv2
import logging
import pyaudio
import pygame
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

logging.basicConfig(filename='logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
siren = r'./co.mp3'

def load_data():
    mask = np.load('with_mask.npy')
    no_mask = np.load('without_mask.npy')

    mask = mask.reshape(200, 50 * 50 * 3)
    no_mask = no_mask.reshape(200, 50 * 50 * 3)

    mix = np.r_[mask, no_mask]
    labels = np.zeros(mix.shape[0])
    labels[200:] = 1.0

    names = {0: 'Mask', 1: 'No Mask'}

    return mix, labels, names

mix, labels, names = load_data()

def split(mix, labels):
    x_train, x_test, y_train, y_test = train_test_split(mix, labels, test_size=0.3)
    return x_test, x_train, y_test, y_train

x_test, x_train, y_test, y_train = split(mix, labels)

def apply_pca(x_train, x_test):
    pca = PCA(n_components=3)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    return pca, x_train_pca, x_test_pca

pca, x_train_pca, x_test_pca = apply_pca(x_train, x_test)

def train_svm(x_train, y_train):
    svm = SVC()
    svm.fit(x_train, y_train)
    return svm

svm_model = train_svm(x_train_pca, y_train)

pygame.mixer.init()  # Initialize the mixer
siren_sound = pygame.mixer.Sound(siren)  # Load the sound file

haar_data = cv2.CascadeClassifier('dataface.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

def detect(svm, names, siren_sound):
    while True:
        flag, img = capture.read()
        if flag:
            faces = haar_data.detectMultiScale(img)
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
                face = img[y:y + h, x:x + w, :]
                face = cv2.resize(face, (50, 50))
                face = face.reshape(1, -1)
                face_pca = pca.transform(face)
                pred = svm.predict(face_pca)
                n = names[int(pred)]
                cv2.putText(img, n, (x, y), font, 1, (244, 250, 250), 2)

                if n == 'Mask':
                    pass
                else:
                    logging.info("Person is not wearing a mask")
                    siren_sound.play()  # Play the sound
            
        
            cv2.imshow('window',img)
            if cv2.waitKey(2) == 27:
                break

# detect(svm_model, names, siren_sound)
def stop():
    capture.release()
    cv2.destroyAllWindows()
    pygame.quit()

'''import frontend
frontend.run(svm_model, pca, names, capture, siren_sound)'''