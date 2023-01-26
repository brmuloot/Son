# -*- coding: utf-8 -*-
"""
Script to learn a model with Scikit-learn.

Created on Mon Oct 24 20:51:47 2022

@author: ValBaron10
"""

import numpy as np
from sklearn import preprocessing
from sklearn import svm
import pickle
from joblib import dump
from features_functions import compute_features
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# LOOP OVER THE SIGNALS

learningFeatures = np.empty((90,71))
learningLabels = np.zeros((90))

for i in range(90):
    # Get an input signal
    file = open("Data/{}".format("signaux"+str(i)), 'rb')
    input_sig = pickle.load(file)

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
    if i < 45:
        learningFeatures[i] = features_vector # 2D np.array with features_vector in it, for each signal
    else:
        learningFeatures[i] = features_vector 
    

    # Store the labels
    if i < 45:
        learningLabels[i] = 0 # Les sinus
    else:
        learningLabels[i] = 1 # Les bruits blancs

# train test split
X_train, X_test, y_train, y_test = train_test_split(learningFeatures, learningLabels, test_size=0.95)

# Encode the class names

labelEncoder = preprocessing.LabelEncoder().fit(y_train)
learningLabelsStd = labelEncoder.transform(y_train)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
learningFeatures_scaled = scaler.transform(X_train)
model.fit(learningFeatures_scaled, learningLabelsStd)

predict = model.predict(X_test)

print(predict, y_test)

accuracy = accuracy_score(y_test, predict)

print("Accuracy:", (accuracy*100),"%")

# Export the scaler and model on disk
dump(scaler, "SCALER")
dump(model, "SVM_MODEL")

