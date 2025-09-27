from tkinter import *
from tkinter import messagebox, simpledialog, filedialog
import numpy as np
import pandas as pd
import os
import pickle
import cv2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from skimage import color
from skimage.feature import graycomatrix, graycoprops
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from tensorflow.keras.models import Sequential, model_from_json

# Global Variables
accuracy = []
precision = []
recall = []
fscore = []
labels = [
    'Chilli___Bacterial_spot', 'Chilli___healthy', 'Cotton___Black_rot', 'Cotton___Esca_(Black_Measles)', 
    'Cotton___healthy', 'Cotton___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Rice___Brownspot', 'Rice___Healthy', 
    'Rice___Leafblast', 'Rice___Leafblight', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]
fertilizers = [
    'Twice in a month,Mancozeb', 'No fertilizers required', 'Thielaviopsis', 'Fungiside', 'No fertilizers required',
    'Micronutrients based Fertilizer', 'Fertilizers with N,P,K', 'No fertilizers required', 'Magnaporthe Oryzae', 
    'Xanthomous Oryzae', 'Balanced Fertilizer(N,P,K)', 'Micronutrients', 'No fertilizers required', 'Pathogen related', 
    'Fungicides Related', 'Balanced Fertilizer(N,P,K)', 'Natural Predators', 'Potassium', 'Weed Control based', 'Organic Mulch'
]

# GUI setup
main = Tk()
main.title("Plant Disease Detection and Fertilizer Recommendation")
main.geometry("1300x1200")
text = Text(main, height=15, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)

# Function to load dataset
def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n\n')
    text.insert(END, "Different Diseases Found in Dataset : " + str(labels) + "\n\n")
    text.insert(END, "Total diseases are : " + str(len(labels)))

# Function to convert image to grayscale
def rgb2gray(image):
    return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

# Function for Green Channel Removal
def remove_green_pixels(image):
    channels_first = image.transpose((2, 0, 1))
    r_channel, g_channel, b_channel = channels_first[0], channels_first[1], channels_first[2]
    mask = np.logical_not(np.logical_and(g_channel > r_channel, g_channel > b_channel))
    channels_first = np.multiply(channels_first, mask)
    return channels_first.transpose(1, 2, 0)

# Feature Extraction Functions
def glcm(image, offsets=[1], angles=[0]):
    single_channel_image = rgb2gray(image) if len(image.shape) > 2 else image
    gclm = greycomatrix(single_channel_image, offsets, angles)
    return gclm

def extract_features(image):
    offsets, angles = [1, 3, 10, 20], [0, np.pi / 4, np.pi / 2]
    image = remove_green_pixels(image)
    gray_image = rgb2gray(image)
    glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
    return np.concatenate([
        np.mean(glcmatrix), np.std(glcmatrix), stats.entropy(glcmatrix.flatten())
    ])

# Model training and evaluation
def calculateMetrics(algorithm, predict, y_test):
    a, p, r, f = accuracy_score(y_test, predict) * 100, precision_score(y_test, predict, average='macro') * 100, recall_score(y_test, predict, average='macro') * 100, f1_score(y_test, predict, average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, f"{algorithm} Accuracy  :  {a}\n{algorithm} Precision : {p}\n{algorithm} Recall : {r}\n{algorithm} FScore : {f}\n\n")

def runSVM():
    global X_train, X_test, y_train, y_test
    if os.path.exists('model/svm.txt'):
        with open('model/svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
    else:
        svm_cls = svm.SVC()
        svm_cls.fit(X_train, y_train)
        with open('model/svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

# GUI components and main event loop
font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Plant Dataset", command=uploadDataset)
uploadButton.place(x=50, y=500)
uploadButton.config(font=font1)

# Initialize the Tkinter window
main.mainloop()
