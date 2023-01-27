# -*- coding: utf-8 -*-
"""
Script to learn a model with Scikit-learn.

Created on Mon Oct 24 20:51:47 2022

@author: ValBaron10
"""

import numpy as np
import matplotlib.pyplot as plt
import wave

from sklearn import preprocessing
from sklearn import svm
from joblib import dump
from features_functions import compute_features
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def load_wave(filepath, divide_factor=1):

  with wave.open(filepath, "rb") as file:
    audio = file.readframes(file.getnframes())

    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2**15
    audio_normalised = audio_as_np_float32 / max_int16

    print(audio_normalised.shape)
    sections = np.array_split(audio_normalised, divide_factor)
    return sections

# l'extension de notre fichier audio ici un fichier wave donc .wav
ext = ".wav"
# chemin pour accèder au dossier depuis le fichier .py
classes_path = ["DataBriefAudio/Pompier/", "DataBriefAudio/Police/"]
classes_path_select = ["DataBriefAudio/Selected/pompier", "DataBriefAudio/Selected/police"]
# On créer des array vides 

divide = 20
obs_tot = 24
obs_class = int(obs_tot/2)

# LOOP OVER THE SIGNALS
learningLabels = []
learningFeatures = False
for i in range(obs_tot):
    # Get an input signal
    if i < obs_class:
        input_signal = load_wave(classes_path_select[0]+"{}".format(" ("+str(i+1)+")"+ext), divide_factor=divide)
    else:
        input_signal = load_wave(classes_path_select[1]+"{}".format(" ("+str(i+1 - 12)+")"+ext), divide_factor=divide)
        
    print(i+1,"/", obs_tot,"\t", round((100 * (i+1) / obs_tot)),"%")

    c = i * divide

    for input_sig in input_signal:

        # Compute the signal in three domains
        sig_sq = input_sig**2
        sig_t = input_sig / np.sqrt(sig_sq.sum())
        sig_f = np.absolute(np.fft.fft(sig_t))
        sig_c = np.absolute(np.fft.fft(sig_f))

        # Compute the features and store them   
        features_list = []
        N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2])
        features_vector = np.array(features_list)[np.newaxis,:]

        # Store the obtained features in a np.arrays
        if type(learningFeatures) == bool:
            learningFeatures = features_vector
            learningLabels.append(0)
        elif i < obs_class :
            learningFeatures = np.append(learningFeatures, features_vector, axis=0) # 2D np.array with features_vector in it, for each signal
            learningLabels.append(0) #pompier
        else:
            learningFeatures = np.append(learningFeatures, features_vector, axis=0)
            learningLabels.append(1) #police


        


# Encode the class names
labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
learningLabelsStd = labelEncoder.transform(learningLabels)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(learningFeatures)
learningFeatures_scaled = scaler.transform(learningFeatures)

# train test split
print(learningFeatures_scaled.shape, learningLabelsStd.shape)
X_train, X_test, y_train, y_test = train_test_split(learningFeatures_scaled, learningLabelsStd, test_size=0.1)

# Training model and predict
model.fit(X_train, y_train)
predict = model.predict(X_test)

# Accuracy 
print(predict, y_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy:", (accuracy*100),"%")

# Matrix confusion
plot_confusion_matrix(model, X_test, y_test) 
plt.show()

# Export the scaler and model on disk
dump(scaler, "SCALER")
dump(model, "SVM_MODEL")