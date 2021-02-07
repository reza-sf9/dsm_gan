import pandas as pd
import numpy as np
import math

def get_mini_batch(size_batch, DATA, shuffling):
	data, times, events = DATA
	m = data.shape[0]  # number of samples
	num_complete_mini_batches = math.floor(m / size_batch)
	mini_batches = []

	''' shuffling data '''
	if shuffling:
		permutation = list(np.random.permutation(m))
		data = data[permutation, :]
		times = times[permutation]
		events = data_mean[permutation]

	''' creating batches '''
	for i in range(0, num_complete_mini_batches):
		mini_batch_data = data[i * size_batch: i * size_batch + size_batch, :]
		mini_batch_times = times[i * size_batch: i * size_batch + size_batch]
		mini_batch_events = events[i * size_batch: i * size_batch + size_batch]


		mini_batch = [mini_batch_data, mini_batch_times, mini_batch_events]
		mini_batches.append(mini_batch)

	''' creating the last non-complete batch '''
	if m % size_batch != 0:
		mini_batch_data = data[num_complete_mini_batches * size_batch: m, :]
		mini_batch_times = times[num_complete_mini_batches * size_batch: m]
		mini_batch_events = events[num_complete_mini_batches * size_batch: m]


		mini_batch = [mini_batch_data, mini_batch_times, mini_batch_events]
		mini_batches.append(mini_batch)

	return mini_batches, len(mini_batches)


def import_dataset(data_mode):
	if 'METABRIC' == data_mode:

		df = pd.read_csv('./data/METABRIC.csv')
		dat1  = df[['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
		times = (df['duration'].values+1)
		events = df['event'].values
		data = dat1.to_numpy()

	if 'SEER' == data_mode:

		df = pd.read_csv('./data/BREAST_onehot_imputed.csv')
		a = df[1:100]

		events = df['label'].values
		times = (df['tte'].values + 1)
		data = df.drop(columns=['tte', 'label', 'id', 'times']).to_numpy(copy=True)

	if 'SEER_sample' == data_mode:

		df = pd.read_csv('./data/SEER_sample_imputed.csv')

		events = df['label'].values
		times = (df['tte'].values + 1)
		data = df.drop(columns=['tte', 'label', 'id', 'times']).to_numpy(copy=True)

	if 'MIMIC_NL' == data_mode:

		df = pd.read_csv('./data/MIMIC_NL.csv')
		events = df['events'].values
		times = (df['times'].values + 1)
		data = df.drop(columns=['events', 'Unnamed: 0', 'times']).to_numpy(copy=True)

	if 'MIMIC_NL_sample' == data_mode:
		df = pd.read_csv('./data/MIMIC_III_sample.csv')
		events = df['events'].values
		times = (df['times'].values + 1)
		data = df.drop(columns=['events', 'Unnamed: 0', 'times']).to_numpy(copy=True)

	return data, events, times
