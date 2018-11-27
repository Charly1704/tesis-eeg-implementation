#!/usr/bin/env bash

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler


# Carga los archivos y los convierte en numpy array
def load_datasets(name):
	df = pd.read_csv(name);	
	# df = StandardScaler().fit_transform(df);
	array = np.array(df)	
	array = array[:,1:]
	return array;

# Aplica un filtro butterworth low_pass a 64 Hz y otro high_pass a los datos
def butterwort_high_pass(data,highcut,order=4,fs=128):
	fs_norm = 0.5 * fs;
	high_fs_norm = highcut / fs_norm		
	low_fs_norm = 64 / fs_norm	
	b,a = signal.butter(order,high_fs_norm,btype='highpass',analog=True)
	high_data = signal.lfilter(b, a, data);	
	c,d = signal.butter(order,low_fs_norm,btype='lowpass',analog=True)
	filtered_data = signal.lfilter(c, d, high_data);

	return filtered_data

## Limpia la señal eliminando el ruido a muy baja frecuencia
def clean_noise(memory,relax, relax_music):
	temp_memory = np.zeros((np.size(memory,0),np.size(memory,1))) ;
	temp_relax = np.zeros((np.size(relax,0),np.size(relax,1)));
	temp_relax_music = np.zeros((np.size(relax_music,0),np.size(relax_music,1)));	

	print(temp_memory.shape)
	print(memory[:,0])
	# temp_memory[:,0] = memory[:,0]
	# temp_relax[:,0] = relax[:,0]
	# temp_relax_music[:,0] = relax_music[:,0]
	print(np.size(memory,1))
	for column in range(0,np.size(memory,1)):
		print(column)
		temp_memory[:,column] = butterwort_high_pass(memory[:,column],1);
	for column1 in range(0,np.size(relax,1)):
		temp_relax[:,column1] = butterwort_high_pass(relax[:,column1],1);
	for column2 in range(0,np.size(relax_music,1)):
		temp_relax_music[:,column2] = butterwort_high_pass(relax_music[:,column2],1);

	return temp_memory, temp_relax, temp_relax_music

# Obtiene la media absoluta
def absolute_mean(data):
	# print(data.shape)
	# print(np.subtract(np.mean(data),data))
	# x = np.full(data.shape,np.mean(data))	
	x = np.sum(data) / len(data);	
	resta = data - x;	
	sumatoria = np.sum(np.absolute(resta))	
	return sumatoria / len(data);

# Parte la totalidad de la señal en muestras de tamaño definido, 
# aplica la transformada de fourier y obtiene la media absoluta de 
# esa conversion. Luego, guarda el valor en un arreglo
def slide_windows(data,samplesPerSecons=128):
	data_length = len(data);	
	windows_number = int(data_length/samplesPerSecons)
	results = np.zeros(windows_number)
	for item in range(0,windows_number):
		if(item == 0):
			# FFT Implementation
			print(samplesPerSecons)			
			# plt.plot(range(0,samplesPerSecons), data[0:samplesPerSecons])
			# plt.xlabel('Samples')
			# plt.ylabel('Data')
			# plt.show()
			fft_vector = fft(data[0:samplesPerSecons]) # Try with np.fft

			n = samplesPerSecons # length of the signal
			k = np.arange(n)
			T = n/samplesPerSecons
			frq = k/T # two sides frequency 
			
			frq = frq[range(int(n /2))] # one side frequency 
			Y = fft_vector;
			Y = Y[range(int(n/2))]

			# plt.plot(frq, abs(Y),'r')
			# plt.xlabel('Freq (Hz)')
			# plt.ylabel('|Y(freq)|')
			# plt.show()		

			results[item] = absolute_mean(fft_vector)

			# results[item] = absolute_mean(data[0:samplesPerSecons])
		else:
			index = item * samplesPerSecons
			# # FFT Implementation
			fft_vector = fft(data[index :index + samplesPerSecons]);
			results[item] = absolute_mean(fft_vector)	
			# results[item] = absolute_mean(data[index :index + samplesPerSecons])	
	return results

# Crea el vector de caracteristicas a partir de fft, corte de ventanas y media absoluta
def fft_abs_mean(data,samplesPerSecons):
	data_length = len(data);	
	windows_number = int(data_length/samplesPerSecons)
	fft_vector = np.zeros((windows_number,np.size(data,1))) 	
	for column in range(0,np.size(data,1)):
		print(column)
		fft_vector[:,column] = slide_windows(data[:,column],samplesPerSecons)
	return fft_vector



memory_dataset = load_datasets('memory_dataset.csv');
relax_dataset = load_datasets('relax_dataset.csv');
relax_music_dataset = load_datasets('relax_music_dataset.csv');


# plt.figure(1)
# plt.subplot(211)
# plt.title('Datos Crudos')
# plt.plot(range(0,256),memory_dataset[0:256,0])

fl_memory, fl_relax, fl_relax_music = clean_noise(memory_dataset,relax_dataset,relax_music_dataset);

# Plot of data before and after butterworth filter 
# plt.figure(2)
# plt.subplot(212)
# plt.title('Datos Filtrados')
# plt.plot(range(0,256),fl_memory[0:256,0])
# plt.show();

size_window = int(input("Size of the window? "))

memory_fft = fft_abs_mean(fl_memory,size_window)
relax_fft = fft_abs_mean(fl_relax,size_window)
relax_music_fft = fft_abs_mean(fl_relax_music,size_window)

memory_fft_df =pd.DataFrame(memory_fft)
relax_fft_df =pd.DataFrame(relax_fft)
relax_music_fft_df =pd.DataFrame(relax_music_fft)

memory_fft_df.to_csv('vector_ftt_abs_mean_memory.csv')
relax_fft_df.to_csv('vector_fft_abs_mean_relax.csv')
relax_music_fft_df.to_csv('vector_fft_abs_mean_relax_music.csv')



# print(memory_fft.shape)
# print(relax_fft.shape)
# print(relax_music_fft.shape)

plt.scatter(memory_fft[0:100,0],memory_fft[0:100,4], label='Memoria')
plt.scatter(relax_fft[0:100,0],relax_fft[0:100,4], label='Relajación')
plt.scatter(relax_music_fft[0:100,0],relax_music_fft[0:100,4], label='Relajación-Musica')
plt.title('Muestra de la distrubución de los datos')
plt.xlabel('AF3')
plt.ylabel('AF4')

plt.legend();
plt.grid(True);
plt.show();




# data_new = clean_noise(memory_dataset[0:128,1],128,0.5,45);
# plt.plot(memory_dataset[0:128:,0],fl_memory[0:128,1])
# plt.show()
# relax_dataset = load_datasets('relax_dataset.csv');
# relax_music_dataset = load_datasets('relax_music_dataset.csv');


# print(memory_dataset.shape)
# print(relax_dataset.shape)
# print(relax_music_dataset.shape)
