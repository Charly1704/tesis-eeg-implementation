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




# Dividir vector en entrenamiento y prueba
def max_data_length(data1,data2):

	if(np.size(data1,0) >= np.size(data2,0)):
		return np.size(data2,0)
	else:
		return np.size(data1,0)

def SVM(dataset,columns=[3,4]):	
	data = np.delete(dataset,5,1)
	targets = dataset[:,5];	

	x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.40, random_state=42)
	
	svclassifier = svm.SVC(C=1,kernel='rbf')
	
	svclassifier.fit(x_train, y_train)
	
	y_pred = svclassifier.predict(x_test)
	
	print(confusion_matrix(y_test,y_pred))
	print(classification_report(y_test, y_pred))




# Entrenar modelo y obtener resultados

def standarize_data(data):
	standar_data = StandardScaler().fit_transform(data);

	return standar_data;


# Imprimir matriz de confusion y reporte de clasificación


# Apartado para impresiones random

# Cargar archivos con el vector de caracteristicas de cada clase
memory_dataset = load_datasets('vector_ftt_abs_mean_memory.csv');
relax_dataset = load_datasets('vector_fft_abs_mean_relax.csv');
relax_music_dataset = load_datasets('vector_fft_abs_mean_relax_music.csv');

# Estandariza los datos a una escala mas uniforme
memory_dataset = standarize_data(memory_dataset); 
relax_dataset = standarize_data(relax_dataset);
relax_music_dataset = standarize_data(relax_music_dataset);

print(memory_dataset.shape)

memory_dataset = np.insert(memory_dataset,5, 1, axis=1)
relax_dataset = np.insert(relax_dataset,5, 2, axis=1)
relax_music_dataset = np.insert(relax_music_dataset,5, 3, axis=1)

print(memory_dataset.shape)



eeg_dataset = np.concatenate((memory_dataset,relax_dataset,relax_music_dataset));

print(eeg_dataset[0:10,:])


num_array = list()
num = 2
for i in range(int(num)):
    n = input("Seleccione el indice correspondiente al electrodo de su eleccion( 0=AF3, 1=T7, 2=Pz, 3=T8, 4=AF4):")
    num_array.append(int(n))


# Ploting the results of corss_validation
print("+++++==MEMORIA-RELAJACION==++++")
SVM(eeg_dataset,num_array)

