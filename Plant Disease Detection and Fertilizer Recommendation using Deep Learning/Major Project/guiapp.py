# Import necessary libraries
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, messagebox, simpledialog, filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
import pickle
import cv2
from skimage import color
from skimage.feature import greycomatrix, greycoprops
import scipy.stats as stats
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from tensorflow.keras.models import Sequential, model_from_json

# Define paths for assets and output
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Global variables
global filename
global X, Y
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test
global cnn

# Define labels and corresponding fertilizers
labels = ['Chilli___Bacterial_spot', 'Chilli___healthy', 'Cotton___Black_rot', 'Cotton___Esca_(Black_Measles)', 'Cotton___healthy', 'Cotton___Leaf_blight_(Isariopsis_Leaf_Spot)',
          'Rice___Brownspot', 'Rice___Healthy', 'Rice___Leafblast', 'Rice___Leafblight', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
          'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
          'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

fertilizers = ['Twice in a month,Mancozeb', 'No fertlizers required', 'Thielaviopsis', 'Fungiside', 'No fertlizers required', 'Micronutrients based Fertilizer',
               'Fertlizers with N,P,K', 'No fertlizers required', 'Magnaporthe Oryzae', 'Xanthomous Oryzae', 'Balanced Fertilizer(N,P,K)', 'Micronutrients',
               'No fertlizers required', 'Pathogen related', 'Fungicides Related', 'Balanced Fertilizer(N,P,K)',
               'Natural Predators', 'Pttasium', 'Weed Control based', 'Organic Mulch']

# Load the pre-trained CNN model
with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    cnn_classifier = model_from_json(loaded_model_json)
json_file.close()
cnn_classifier.load_weights("model/model_weights.h5")


########################################################################################################################
# Function to remove green pixels from the image
def remove_green_pixels(image):
    channels_first = channels_first_transform(image)
    r_channel = channels_first[0]
    g_channel = channels_first[1]
    b_channel = channels_first[2]
    mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
    channels_first = np.multiply(channels_first, mask)
    image = channels_first.transpose(1, 2, 0)
    return image

# Function to convert RGB image to LAB color space
def rgb2lab(image):
    return color.rgb2lab(image)

# Function to convert RGB image to grayscale
def rgb2gray(image):
    return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

# Function to compute GLCM (Gray Level Co-occurrence Matrix)
def glcm(image, offsets=[1], angles=[0], squeeze=False):
    single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
    gclm = greycomatrix(single_channel_image, offsets, angles)
    return np.squeeze(gclm) if squeeze else gclm

# Function to compute histogram features with bucket count
def histogram_features_bucket_count(image):
    image = channels_first_transform(image).reshape(3, -1)
    r_channel = image[0]
    g_channel = image[1]
    b_channel = image[2]
    r_hist = np.histogram(r_channel, bins=26, range=(0, 255))[0]
    g_hist = np.histogram(g_channel, bins=26, range=(0, 255))[0]
    b_hist = np.histogram(b_channel, bins=26, range=(0, 255))[0]
    return np.concatenate((r_hist, g_hist, b_hist))

# Function to compute histogram features
def histogram_features(image):
    color_histogram = np.histogram(image.flatten(), bins=255, range=(0, 255))[0]
    return np.array([
        np.mean(color_histogram),
        np.std(color_histogram),
        stats.entropy(color_histogram),
        stats.kurtosis(color_histogram),
        stats.skew(color_histogram),
        np.sqrt(np.mean(np.square(color_histogram)))
    ])

# Function to compute texture features
def texture_features(full_image, offsets=[1], angles=[0], remove_green=True):
    image = remove_green_pixels(full_image) if remove_green else full_image
    gray_image = rgb2gray(image)
    glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
    return glcm_features(glcmatrix)

# Function to compute GLCM features
def glcm_features(glcm):
    return np.array([
        greycoprops(glcm, 'correlation'),
        greycoprops(glcm, 'contrast'),
        greycoprops(glcm, 'energy'),
        greycoprops(glcm, 'homogeneity'),
        greycoprops(glcm, 'dissimilarity'),
    ]).flatten()

# Function to transform image to channels first format
def channels_first_transform(image):
    return image.transpose((2, 0, 1))

# Function to extract features from the image
def extract_features(image):
    offsets = [1, 3, 10, 20]
    angles = [0, np.pi / 4, np.pi / 2]
    channels_first = channels_first_transform(image)
    return np.concatenate((
        texture_features(image, offsets=offsets, angles=angles),
        texture_features(image, offsets=offsets, angles=angles, remove_green=False),
        histogram_features_bucket_count(image),
        histogram_features(channels_first[0]),
        histogram_features(channels_first[1]),
        histogram_features(channels_first[2]),
    ))

# Function to get the index of a label
def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

# Function to upload dataset
def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    entry_1.delete('1.0', tk.END)
    entry_1.insert(tk.END, filename + ' Loaded\n\n')
    entry_1.insert(tk.END, "Different Diseases Found in Dataset : " + str(labels) + "\n\n")
    entry_1.insert(tk.END, "Total diseases are : " + str(len(labels)))

# Function to extract features from the dataset
def featuresExtraction():
    global filename
    global X, Y
    global X_train, X_test, y_train, y_test
    entry_1.delete('1.0', tk.END)
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root + "/" + directory[j])
                    img = cv2.resize(img, (64, 64))
                    class_label = getID(name)
                    features = extract_features(img)
                    Y.append(class_label)
                    X.append(features)
                    print(name + " " + root + "/" + directory[j] + " " + str(features.shape) + " " + str(class_label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X", X)
        np.save("model/Y", Y)
    X = X.astype('float32')
    X = X / 255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    entry_1.insert(tk.END, "Extracted GLCM & Texture Features : " + str(X[0]) + "\n\n")
    entry_1.insert(tk.END, "Total images found in dataset : " + str(X.shape[0]) + "\n\n")
    entry_1.insert(tk.END, "Dataset train & test split. 80% dataset images used for training and 20% for testing\n\n")
    entry_1.insert(tk.END, "80% training images : " + str(X_train.shape[0]) + "\n\n")
    entry_1.insert(tk.END, "20% training images : " + str(X_test.shape[0]) + "\n\n")

# Function to calculate metrics for the model
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    entry_1.insert(tk.END, algorithm + " Accuracy  :  " + str(a) + "\n")
    entry_1.insert(tk.END, algorithm + " Precision : " + str(p) + "\n")
    entry_1.insert(tk.END, algorithm + " Recall    : " + str(r) + "\n")
    entry_1.insert(tk.END, algorithm + " FScore    : " + str(f) + "\n\n")

# Function to run SVM model
def runSVM():
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    entry_1.delete('1.0', tk.END)

    svm_cls = svm.SVC(probability=True)
    svm_cls.fit(X, Y)
    with open('model/svm.txt', 'wb') as file:
        pickle.dump(svm_cls, file)
    file.close()

    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

# Function to run DNN model
def runDNN():
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision, recall, fscore
    if os.path.exists('model/fcnn.txt'):
        with open('model/fcnn.txt', 'rb') as file:
            fcnn_cls = pickle.load(file)
        file.close()
    else:
        fcnn_cls = MLPClassifier(max_iter=100)
        fcnn_cls.fit(X, Y)
        with open('model/fcnn.txt', 'wb') as file:
            pickle.dump(fcnn_cls, file)
        file.close()
    predict = fcnn_cls.predict(X_test)
    calculateMetrics("DNN", predict, y_test)

# Function to run CNN model
def runCNN():
    global X_train, X_test, y_train, y_test, X, Y, cnn
    global accuracy, precision, recall, fscore
    Y1 = to_categorical(Y)
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y1, test_size=0.2)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        json_file.close()
        cnn.load_weights("model/model_weights.h5")
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(32, 1, 1, input_shape=(XX.shape[1], XX.shape[2], XX.shape[3]), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(1, 1)))
        cnn.add(Convolution2D(32, 1, 1, activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(1, 1)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim=256, activation='relu'))
        cnn.add(Dense(output_dim=Y1.shape[1], activation='softmax'))
        cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = cnn.fit(XX, Y1, batch_size=12, epochs=10, shuffle=True, verbose=2)
        cnn.save_weights('model/model_weights.h5')
        model_json = cnn.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn.summary())
    predict = cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    calculateMetrics("Propose CNN", predict, y_test)

# Function to plot the graph of metrics
def graph():
    df = pd.DataFrame([['SVM', 'Accuracy', accuracy[0]], ['SVM', 'Precision', precision[0]], ['SVM', 'Recall', recall[0]], ['SVM', 'FScore', fscore[0]],
                       ['DNN', 'Accuracy', accuracy[1]], ['DNN', 'Precision', precision[1]], ['DNN', 'Recall', recall[1]], ['DNN', 'FScore', fscore[1]],
                       ['Propose CNN', 'Accuracy', accuracy[2]], ['Propose CNN', 'Precision', precision[2]], ['Propose CNN', 'Recall', recall[2]], ['Propose CNN', 'FScore', fscore[2]],
                       ], columns=['Parameters', 'Algorithms', 'Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

# Function to predict the disease and recommend fertilizer
def predict():
    global cnn
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    test = []
    img = cv2.resize(img, (64, 64))
    features = extract_features(img)
    test.append(features)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test / 255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = cnn.predict(test)
    predict = np.argmax(predict)

    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 400))

    # Define the font and color for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color
    font_thickness = 2

    # Define the background color for the text
    bg_color = (0, 0, 0)  # Black color

    # Calculate the size of the text to create a background rectangle
    text_size1 = cv2.getTextSize('Plant Disease Predicted as : ' + labels[predict], font, font_scale, font_thickness)[0]
    text_size2 = cv2.getTextSize('Recommended Fertilizer is : ' + fertilizers[predict], font, font_scale, font_thickness)[0]

    # Draw the background rectangles
    cv2.rectangle(img, (10, 10), (10 + text_size1[0], 10 + text_size1[1] + 10), bg_color, -1)
    cv2.rectangle(img, (10, 50), (10 + text_size2[0], 50 + text_size2[1] + 10), bg_color, -1)

    # Put the text on the image
    cv2.putText(img, 'Plant Disease Predicted as : ' + labels[predict], (10, 30), font, font_scale, font_color, font_thickness)
    cv2.putText(img, 'Recommended Fertilizer is : ' + fertilizers[predict], (10, 70), font, font_scale, font_color, font_thickness)

    cv2.imshow('Plant Disease Prediction', img)
    cv2.waitKey(0)


#####################################################################################################################################
# Create the main window
window = Tk()

# Set the window size and background color
window.geometry("1205x754")
window.configure(bg="#FFFFFF")

# Create a canvas for the GUI elements
canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=754,
    width=1205,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    602.0,
    36.0,
    image=image_image_1
)

canvas.create_rectangle(
    0.0,
    67.0,
    1205.0,
    754.0,
    fill="#A5D6A7",
    outline=""
)

# Create buttons for various functionalities
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=uploadDataset,
    relief="flat"
)
button_1.place(
    x=780.0,
    y=81.0,
    width=417.0,
    height=103.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=featuresExtraction,
    relief="flat"
)
button_2.place(
    x=780.0,
    y=184.0,
    width=417.0,
    height=123.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=runSVM,
    relief="flat"
)
button_3.place(
    x=780.0,
    y=307.0,
    width=417.0,
    height=92.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=runDNN,
    relief="flat"
)
button_4.place(
    x=780.0,
    y=399.0,
    width=417.0,
    height=99.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=runCNN,
    relief="flat"
)
button_5.place(
    x=780.0,
    y=498.0,
    width=417.0,
    height=102.0
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=graph,
    relief="flat"
)
button_6.place(
    x=5.0,
    y=620.0,
    width=405.0,
    height=110.0
)

button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=predict,
    relief="flat"
)
button_7.place(
    x=405.0,
    y=600.0,
    width=792.0,
    height=143.0
)

# Create a text entry for displaying output
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    382.5,
    342.0,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#0E0000",
    fg="#FFFFFF",  # Set text color to white
    highlightthickness=0,
    font=("Arial", 12, "bold")  # Increase font size and use bold style
)
entry_1.place(
    x=31.0,
    y=96.0,
    width=703.0,
    height=490.0
)

# Make the window non-resizable
window.resizable(False, False)

# Start the main loop of the GUI application
window.mainloop()