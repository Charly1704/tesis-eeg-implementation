import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft


# Carga los archivos y los convierte en numpy array
def load_datasets(name):
	df = pd.read_csv(name);	
	array = np.array(df)
	# array = array[:,1:]
	return array;

# Aplica un filtro butterworth high_pass a los datos
def butterwort_high_pass(data,highcut,order=4,fs=128):
	fs_norm = 0.5 * fs;
	high_fs_norm = highcut / fs_norm	
	b,a = signal.butter(order,high_fs_norm,btype='highpass',analog=True)
	high_data = signal.lfilter(b, a, data);	
	# c,d = signal.butter(order,low_fs_norm,btype='lowpass',analog=True)
	# filtered_data = signal.lfilter(c, d, high_data);

	return high_data

## Limpia la señal eliminando el ruido a muy baja frecuencia
def clean_noise(memory,relax, relax_music):
	temp_memory = np.zeros((np.size(memory,0),np.size(memory,1))) ;
	temp_relax = np.zeros((np.size(relax,0),np.size(relax,1)));
	temp_relax_music = np.zeros((np.size(relax_music,0),np.size(relax_music,1)));

	print(temp_memory.shape)
	print(temp_relax.shape)
	print(temp_relax_music.shape)

	temp_memory[:,0] = memory[:,0]
	temp_relax[:,0] = relax[:,0]
	temp_relax_music[:,0] = relax_music[:,0]

	for column in range(1,np.size(memory[:,1:],1)+1):
		
		temp_memory[:,column] = butterwort_high_pass(memory[:,column],1);
	for column1 in range(1,np.size(relax[:,1:],1)+1):
		temp_relax[:,column1] = butterwort_high_pass(relax[:,column1],1);
	for column2 in range(1,np.size(relax_music[:,1:],1)+1):
		temp_relax_music[:,column2] = butterwort_high_pass(relax_music[:,column2],1);

	return temp_memory, temp_relax, temp_relax_music

# Obtiene la media absoluta
def absolute_mean(data):
	return np.sum(np.absolute(data)) / len(data);

# Parte la totalidad de la señal en muestras de tamaño definido, 
# aplica la transformada de fourier y obtiene la media absoluta de 
# esa conversion. Luego, guarda el valor en un arreglo
def slide_windows(data,samplesPerSecons=128):
	data_length = len(data);	
	windows_number = int(data_length/samplesPerSecons)
	results = np.zeros(windows_number)
	for item in range(0,windows_number):
		if(item == 0):
			fft_vector = fft(data[0:samplesPerSecons])
			results[item] = absolute_mean(fft_vector)
		else:
			index = item * samplesPerSecons
			fft_vector = fft(data[index :index + samplesPerSecons]);
			results[item] = absolute_mean(fft_vector)	
	return results

# Crea el vector de caracteristicas a partir de fft, corte de ventanas y media absoluta
def fft_abs_mean(data,samplesPerSecons):
	data_length = len(data);	
	windows_number = int(data_length/samplesPerSecons)
	fft_vector = np.zeros((windows_number,np.size(data,1))) 	
	for column in range(1,np.size(data[:,1:],1)+1):
		fft_vector[:,column] = slide_windows(data[:,column],samplesPerSecons)
	return fft_vector



memory_dataset = load_datasets('memory_dataset.csv');
relax_dataset = load_datasets('relax_dataset.csv');
relax_music_dataset = load_datasets('relax_music_dataset.csv');

fl_memory, fl_relax, fl_relax_music = clean_noise(memory_dataset,relax_dataset,relax_music_dataset);

memory_fft = fft_abs_mean(fl_memory,128)
relax_fft = fft_abs_mean(fl_relax,128)
relax_music_fft = fft_abs_mean(fl_relax_music,128)

memory_fft_df =pd.DataFrame(memory_fft)
relax_fft_df =pd.DataFrame(relax_fft)
relax_music_fft_df =pd.DataFrame(relax_music_fft)

memory_fft_df.to_csv('vector_ftt_abs_mean_memory.csv')
relax_fft_df.to_csv('vector_fft_abs_mean_relax.csv')
relax_music_fft_df.to_csv('vector_fft_abs_mean_relax_music.csv')



# print(memory_fft.shape)
# print(relax_fft.shape)
# print(relax_music_fft.shape)

# plt.scatter(memory_fft[0:400,1],memory_fft[0:400,4])
# plt.scatter(relax_fft[0:400,1],relax_fft[0:400,4])
# plt.scatter(relax_music_fft[0:400,1],relax_music_fft[0:400,4])



# plt.show();




# data_new = clean_noise(memory_dataset[0:128,1],128,0.5,45);
# plt.plot(memory_dataset[0:128:,0],fl_memory[0:128,1])
# plt.show()
# relax_dataset = load_datasets('relax_dataset.csv');
# relax_music_dataset = load_datasets('relax_music_dataset.csv');


# print(memory_dataset.shape)
# print(relax_dataset.shape)
# print(relax_music_dataset.shape)
