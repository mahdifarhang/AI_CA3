import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import math
from time import time

num_of_training_data = 4200

def getting_train_data(file_name):

	temp = pd.read_csv(file_name)
	data = []
	deleting_words = list(stopwords.words('english'))
	deleting_words.extend(['.', ',', '..', '...', '?'])
	ps = PorterStemmer()

	for i in range(len(temp)):
		words = nltk.word_tokenize(temp['text'][i])
		filtered_words = []
		for word in words:
			if (ps.stem(word) not in deleting_words) and (ps.stem(word) not in filtered_words):
				filtered_words.append(ps.stem(word))
		# filtered_words = list(dict.fromkeys(filtered_words))
		data.append([temp['type'][i], filtered_words])
	return data

def possibility_of_word(data, word):
	num_of_spam = 0
	num_of_ham = 0
	num_of_word_in_spams = 0
	num_of_word_in_hams = 0
	for item in data:
		if (item[0] == 'spam'):
			num_of_spam += 1
			num_of_word_in_spams += 1 if (word in item[1]) else 0
		else:
			num_of_ham += 1
			num_of_word_in_hams += 1 if (word in item[1]) else 0
	return float((num_of_word_in_spams if num_of_word_in_spams != 0 else 0.1) / num_of_spam), float((num_of_word_in_hams if num_of_word_in_hams != 0 else 0.1) / num_of_ham)

def train(data):
	spams = {}
	hams = {}
	num_of_spam_in_size = [0] * 80
	num_of_ham_in_size = [0] * 80
	i = 0
	for sentence in data:
		for word in sentence[1]:
			if word not in spams:
				a, b = possibility_of_word(data[0:num_of_training_data], word)
				spams[word] = a
				hams[word] = b
		if (i > num_of_training_data):
			continue
		if (sentence[0] == 'spam'):
			num_of_spam_in_size[len(sentence[1])] += 1
		else:
			num_of_ham_in_size[len(sentence[1])] += 1
		i += 1
	return spams, hams, num_of_spam_in_size, num_of_ham_in_size

def guess_type_of_sentence(spams, hams, words, num_of_spams, num_of_hams, spam_sizes, ham_sizes):
	
	total = num_of_spams + num_of_hams
	p_all_spams = math.log(num_of_spams / total, 2)
	p_all_hams = math.log(num_of_hams / total, 2)
	for word in words:
		p_all_spams += math.log(spams[word], 2)
		p_all_hams += math.log(hams[word], 2)
	temp = len(words)
	p_all_spams += temp / 3 * math.log(((spam_sizes[temp] if spam_sizes[temp] != 0 else 0.1) / num_of_spams), 2)
	p_all_hams += temp / 3 * math.log(((ham_sizes[temp] if ham_sizes[temp] != 0 else 0.1) / num_of_hams), 2)
	if (p_all_spams > p_all_hams):	
		return 'spam'
	return 'ham'

def test(data, spams, hams, spam_sizes, ham_sizes):
	num_of_spam = 0
	num_of_ham = 0
	for item in data:
		if (item[0] == 'spam'):
			num_of_spam += 1
		else:
			num_of_ham += 1

	result = []
	for i in range(num_of_training_data, len(data)):
		result.append(guess_type_of_sentence(spams, hams, data[i][1], num_of_spam, num_of_ham, spam_sizes, ham_sizes))
	
	total = len(result)
	true_spam = 0
	true_ham = 0
	false_spam = 0
	false_ham = 0

	for i in range(total):
		if (data[num_of_training_data + i][0] == 'spam'):
			if (result[i] == 'spam'):
				true_spam += 1
			else:
				false_spam += 1
		else:
			if (result[i] == 'ham'):
				true_ham += 1
			else:
				false_ham += 1
	print('number of test data = ', total)
	print('Recall = ', true_spam / (true_spam + false_spam))
	print('Precision = ', true_spam / (true_spam + false_ham))
	print('Accuracy = ', (true_spam + true_ham) / total)

t1 = time()
data = getting_train_data('train_test.csv')
spams, hams, spam_sizes, ham_sizes = train(data)
t2 = time()
print('training time ', t2 - t1)

test(data, spams, hams, spam_sizes, ham_sizes)
