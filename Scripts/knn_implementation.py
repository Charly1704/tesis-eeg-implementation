import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from vector_fft_media_absoluta import load_datasets
from sklearn.metrics import confusion_matrix

## Utilizar arbol de desición para obtener las mejores caracteristicas y PCA

## Obtiene el error de clasificación utilizando diferentes numero 
# de vecinos en el clasificador
def cross_validation(x_train,y_train):
	# creating odd list of K for KNN
	myList = list(range(1,50))

	# subsetting just the odd ones
	neighbors =[x for x in myList if x % 2 != 0];
	print(neighbors)

	# empty list that will hold cv scores
	cv_scores = []

	# perform 10-fold cross validation
	for k in neighbors:
	    knn = KNeighborsClassifier(n_neighbors=k,weights='distance',metric='minkowski',p=1)
	    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
	    cv_scores.append(scores.mean())
	# changing to misclassification error
	MSE = [1 - x for x in cv_scores]

	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	print("The optimal number of neighbors is {}".format(optimal_k))

	return MSE, neighbors

# Realiza el entrenamiento y la clasificación utilizando el clasificador KNN
def KNN(class_1,class_2,length,apply_cross_validation=False,neighbors=17,columns=[3,5]):
	target_class_1 = np.repeat(1,length);
	target_class_2 = np.repeat(2,length);	
	targets = np.concatenate((target_class_1,target_class_2));
	eeg_dataset = np.concatenate((class_1[0:length,columns], class_2[0:length,columns]))

	x_train, x_test, y_train, y_test = train_test_split(eeg_dataset, targets, test_size=0.40, random_state=42)
	if(apply_cross_validation):
		MSE, neighbors_list = cross_validation(x_train, y_train);
	knn = KNeighborsClassifier(n_neighbors=neighbors,weights='distance',metric='minkowski',p=1)
	knn.fit(x_train, y_train)

	pred = knn.predict(x_test)
	if(apply_cross_validation):
		print(confusion_matrix(y_test,pred))
		return accuracy_score(y_test,pred),MSE, neighbors_list
	else:
		print(confusion_matrix(y_test,pred))
		return accuracy_score(y_test,pred)

## Grafica los resultados del cross_validation
def plot_cross_validation(MSE,neigbors):
	# plot misclassification error vs k
	print("The data gets an score of {}".format(score))
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

# Carga de los vectores de caracteristicas
memory_dataset = load_datasets('vector_ftt_abs_mean_memory.csv');
relax_dataset = load_datasets('vector_fft_abs_mean_relax.csv');
relax_music_dataset = load_datasets('vector_fft_abs_mean_relax_music.csv');

# Getting the mean of the accuracy score
#accuracy_mean_score(50,memory_dataset,relax_dataset)

# Ploting the results of corss_validation
# score, MSE, neighbors = KNN(memory_dataset,relax_dataset,1488,True);
# plot_cross_validation(MSE, neighbors)

print(KNN(memory_dataset,relax_music_dataset,3620,False,1))
print(KNN(memory_dataset,relax_dataset,1488,False,1))
print(KNN(relax_dataset,relax_music_dataset,1488,False,1))




# print(memory_dataset.shape)
# print(relax_dataset.shape)
# print(relax_music_dataset.shape)