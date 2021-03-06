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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



## Obtiene el error de clasificación utilizando diferentes numero 
# de vecinos en el clasificador
def cross_validation(x_train,y_train, x_test, y_test):
	# creating odd list of K for KNN
	myList = list(range(1,100))

	# subsetting just the odd ones
	neighbors =[x for x in myList if x % 2 != 0];	

	# empty list that will hold cv scores
	cv_scores = []
	final_results = [];	

	targets = np.concatenate((y_train, y_test))
	data = np.concatenate((x_train, x_test))

	# perform 10-fold cross validation
	for k in neighbors:
	    knn = KNeighborsClassifier(n_neighbors=k,weights='distance',metric='euclidean',p=2)
	    scores = cross_val_score(knn, data, targets, cv=10, scoring='accuracy')
	    # print("Scores Length: {}".format(len(scores)))	    
	    cv_scores.append(scores.mean())
	# changing to misclassification error
	MSE = [1 - x for x in cv_scores]	
	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	
	print("The optimal number of neighbors is {}".format(optimal_k))

	for k in range(0,100):
		knn = KNeighborsClassifier(n_neighbors=optimal_k,weights='distance',metric='euclidean',p=2)
		scores = cross_val_score(knn, data, targets, cv=10, scoring='accuracy')			

		final_results.append(scores.mean())
	final_results = np.array(final_results)
	print("Number of results {}".format(np.size(final_results,0)))
	print("The best result after 100 runs and 10-fold cross validation was {}%".format(np.around(final_results.mean() * 100, decimals=4)));


	return MSE, neighbors

# Realiza el entrenamiento y la clasificación utilizando el clasificador KNN
def KNN(class_1,class_2,apply_cross_validation=False,neighbors=17,columns=[3,4]):
	length = max_data_length(class_1,class_2);
	
	target_class_1 = np.repeat(1,length);
	target_class_2 = np.repeat(2,length);	
	targets = np.concatenate((target_class_1,target_class_2));
	eeg_dataset = np.concatenate((class_1[0:length,columns], class_2[0:length,columns]))

	x_train, x_test, y_train, y_test = train_test_split(eeg_dataset, targets, test_size=0.40, random_state=42)
	if(apply_cross_validation):
		MSE, neighbors_list = cross_validation(x_train, y_train, x_test, y_test);
	knn = KNeighborsClassifier(n_neighbors=neighbors,weights='distance',metric='euclidean',p=2)
	knn.fit(x_train, y_train)

	pred = knn.predict(x_test)
	if(apply_cross_validation):
		print(confusion_matrix(y_test,pred))
		print(classification_report(y_test, pred))
		return accuracy_score(y_test,pred),MSE, neighbors_list
	else:
		print(confusion_matrix(y_test,pred))
		return accuracy_score(y_test,pred)

## Grafica los resultados del cross_validation
def plot_cross_validation(MSE,neigbors):
	# plot misclassification error vs k	
	plt.plot(neighbors, MSE)
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('Misclassification Error')
	plt.show()

## Función que realiza una numero definido de corridas del clasificador
# para obtner un promedio como resultdo final
def accuracy_mean_score(runs_number,data1,data2,length=1488):
	score_sum = 0
	for x in range(1,runs_number):
		score_sum += KNN(data1,data2,length);
	print("The Real Score for memory and relax is {}".format(score_sum/x))

# def standarize_data(data):
# 	standar_data = StandardScaler().fit_transform(data);

# 	return standar_data;

def pca_implementation(x):
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)
	principalDf = pd.DataFrame(data = principalComponents, columns =['Principal Component 1','Principal Component 2'])
	principalNumpy = np.array(principalDf);

	return principalNumpy

def max_data_length(data1,data2):
	if(np.size(data1,0) >= np.size(data2,0)):
		return np.size(data2,0)
	else:
		return np.size(data1,0)


# Carga de los vectores de caracteristicas
memory_dataset = load_datasets('vector_ftt_abs_mean_memory.csv');
relax_dataset = load_datasets('vector_fft_abs_mean_relax.csv');
relax_music_dataset = load_datasets('vector_fft_abs_mean_relax_music.csv');

# Estandariza los datos a una escala mas uniforme
# memory_dataset = standarize_data(memory_dataset[:,2:]); 
# relax_dataset = standarize_data(relax_dataset[:,2:]);
# relax_music_dataset = standarize_data(relax_music_dataset[:,2:]);



# memory_pca = pca_implementation(memory_dataset); 
# relax_pca = pca_implementation(relax_dataset);
# relax_music_pca = pca_implementation(relax_music_dataset);


num_array = list()
num = 2
for i in range(int(num)):
    n = input("Seleccione el indice correspondiente al electrodo de su eleccion( 0=AF3, 1=T7, 2=Pz, 3=T8, 4=AF4):")
    num_array.append(int(n))


# Getting the mean of the accuracy score
#accuracy_mean_score(50,memory_dataset,relax_dataset)

# Ploting the results of corss_validation
print("+++++==MEMORIA-RELAJACION==++++")
score, MSE, neighbors = KNN(memory_dataset,relax_dataset,True,33,num_array);
plot_cross_validation(MSE, neighbors)

print("+++++==MEMORIA-RELAJACION_MUSICA==++++")
score, MSE, neighbors = KNN(memory_dataset,relax_music_dataset,True,49,num_array);
plot_cross_validation(MSE, neighbors)

print("+++++==RELAJACION-RELAJACION_MUSICA==++++")
score, MSE, neighbors = KNN(relax_dataset,relax_music_dataset,True,23,num_array);
plot_cross_validation(MSE, neighbors)

# Results using PCA Coeficients
# score, MSE, neighbors = KNN(memory_pca,relax_music_pca,True,49);
# plot_cross_validation(MSE, neighbors)

# score, MSE, neighbors = KNN(memory_pca,relax_pca,True,33);
# plot_cross_validation(MSE, neighbors)

# score, MSE, neighbors = KNN(relax_pca,relax_music_pca,True,23);
# plot_cross_validation(MSE, neighbors)

# print(KNN(memory_pca,relax_music_pca,False,13,[0,1]))
# print(KNN(memory_pca,relax_pca,False,13,[0,1]))
# print(KNN(relax_pca,relax_music_pca,False,13,[0,1]))




# print(memory_pca.shape)
# print(relax_dataset.shape)
# print(relax_music_dataset.shape)