import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


data = pd.read_csv('train_test.csv')




num = 0
for i in range(len(data)):
	if data['type'][i] == 'ham':
		num += 1
print(num)






# data = "All the works and no play makes jack a dull boy, all works and no playing."

# phrases = nltk.sent_tokenize(data)
# words = nltk.word_tokenize(data)

# stopWords = set(stopwords.words('english'))
# filtered_words = []

# for word in words:
# 	if word not in stopWords:
# 		filtered_words.append(word)

# ps = PorterStemmer()
# for word in filtered_words:
# 	print(ps.stem(word))


# print(filtered_words)

