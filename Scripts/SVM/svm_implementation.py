#!/usr/bin/env bash

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from vector_fft_media_absoluta import load_datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def cross_validation(x_train, y_train, x_test, y_test,kernel):
	cv_scores = []

	targets = np.concatenate((y_train, y_test))
	data = np.concatenate((x_train, x_test))

	for k in range(0,100):
		svclassifier = svm.SVC(kernel=kernel, gamma=10, class_weight="balanced", degree=3, decision_function_shape='ovo')
		scores = cross_val_score(svclassifier, data, targets, cv=10, scoring='accuracy')
		cv_scores.append(scores.mean())
	cv_scores = np.array(cv_scores)
	print("<<< USED KERNEL {} >>>".format(kernel))
	print("The best result after 100 runs and 10-fold cross validation was {}%".format(np.around(cv_scores.mean() * 100, decimals=4)));


def max_data_length(data1,data2):

	if(np.size(data1,0) >= np.size(data2,0)):
		return np.size(data2,0)
	else:
		return np.size(data1,0)

#Entrenamiento y evaluación del modelo
def SVM(dataset):	
	data = np.delete(dataset,5,1)
	targets = dataset[:,5];	


	x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.40, random_state=42)

	for fig_num, kernel in enumerate(('linear', 'rbf')):
		cross_validation(x_train,y_train,x_test,y_test,kernel)
		svclassifier = svm.SVC(kernel=kernel, gamma=10, class_weight="balanced",decision_function_shape='ovo', degree=3)
	
		svclassifier.fit(x_train, y_train)
		
		y_pred = svclassifier.predict(x_test)
		
		print(confusion_matrix(y_test,y_pred))
		print(classification_report(y_test, y_pred))
		# print(accuracy_score(y_test,y_pred))
		
	




# Estandarización de datos con StandardScaler
def standarize_data(data):
	standar_data = StandardScaler().fit_transform(data);

	return standar_data;

def pca_implementation(x):
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data = principalComponents, columns =['Principal Component 1','Principal Component 2'])
	principalNumpy = np.array(principalDf);

	return principalNumpy




# Cargar archivos con el vector de caracteristicas de cada clase
memory_dataset = load_datasets('vector_ftt_abs_mean_memory.csv');
relax_dataset = load_datasets('vector_fft_abs_mean_relax.csv');
relax_music_dataset = load_datasets('vector_fft_abs_mean_relax_music.csv');


# plt.title('Datos sin normalizar')
# plt.xlabel('Time')
# plt.ylabel('AF3')
# plt.plot(range(0,256),memory_dataset[:256,1])
# plt.show()

# print("<<< Vector de características >>>")
# print(memory_dataset[0:5,:])

# Estandariza los datos a una escala mas uniforme
memory_dataset = standarize_data(memory_dataset); 
relax_dataset = standarize_data(relax_dataset);
relax_music_dataset = standarize_data(relax_music_dataset);


# plt.title('Datos normalizados')
# plt.xlabel('Time')
# plt.ylabel('AF3')
# plt.plot(range(0,256),memory_dataset[:256,1])
# plt.show();

# print("<<< Datos normalizados >>>")
# print(memory_dataset[0:5,:])

# Implementación de PCA
# memory_dataset = pca_implementation(memory_dataset); 
# relax_dataset = pca_implementation(relax_dataset);
# relax_music_dataset = pca_implementation(relax_music_dataset);

# print(memory_dataset.shape)

memory_dataset = np.insert(memory_dataset,5, 1, axis=1)
relax_dataset = np.insert(relax_dataset,5, 2, axis=1)
relax_music_dataset = np.insert(relax_music_dataset,5, 3, axis=1)

# print(memory_dataset.shape)



eeg_dataset = np.concatenate((memory_dataset,relax_dataset,relax_music_dataset));


# print(eeg_dataset[0:10,:])


# Ploting the results of corss_validation
print("+++++==MEMORIA-RELAJACION-RELAJACION_MEMORIA==++++")
SVM(eeg_dataset)

