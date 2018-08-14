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

def SVM(class_1,class_2,kernel='linear',columns=[3,4]):
	length = max_data_length(class_1,class_2)
	print("La clase de mayor tamaño tiene: {0}".format(length))
	target_class_1 = np.repeat(1,length);
	target_class_2 = np.repeat(2,length);	
	targets = np.concatenate((target_class_1,target_class_2));
	eeg_dataset = np.concatenate((class_1[0:length,columns], class_2[0:length,columns]))
	print("Se crearon los vectores")

	x_train, x_test, y_train, y_test = train_test_split(eeg_dataset, targets, test_size=0.40, random_state=42)
	print("Se crearon los destos")
	svclassifier = svm.SVC(kernel='rbf')
	print("Entrenando modelo")
	svclassifier.fit(x_train, y_train)
	print("Obteniendo predicciones")
	y_pred = svclassifier.predict(x_test)
	print("Imprimiendo matrices")
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

num_array = list()
num = 2
for i in range(int(num)):
    n = input("Seleccione el indice correspondiente al electrodo de su eleccion( 0=AF3, 1=T7, 2=Pz, 3=T8, 4=AF4):")
    num_array.append(int(n))

kernel = input("Especifique EL tipo de kernel:");


# Ploting the results of corss_validation
print("+++++==MEMORIA-RELAJACION==++++")
SVM(memory_dataset,relax_dataset,kernel,num_array)


print("+++++==MEMORIA-RELAJACION_MUSICA==++++")
SVM(memory_dataset,relax_music_dataset,kernel,num_array)


print("+++++==RELAJACION-RELAJACION_MUSICA==++++")
SVM(relax_dataset,relax_music_dataset,kernel,num_array)
