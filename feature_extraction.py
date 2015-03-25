import numpy as np
import nltk

with open("movie-reviews-sentiment.tsv") as file:
	file_content = np.array([line.strip().decode('utf-8').split('\t') for line in file])

classes = file_content[:,0]
instances = file_content[:,1:]
vocab = set()
for i in instances:
	tokens = nltk.word_tokenize(i[0])
	for t in tokens:
		vocab.add(t)
vocab = list(vocab)


matrix = []
f = open('myfile','w')
for sentence in enumerate(instances):
	tokens = nltk.word_tokenize(sentence[1][0])
	vector = [0] * len(vocab)
	for t in tokens:
		vector[vocab.index(t)] += 1
	matrix.append(vector)

f.write(matrix)
f.close()

