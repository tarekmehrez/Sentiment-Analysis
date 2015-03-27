import numpy as np
import nltk
import logging


class Corpus(object):

	def __init__(self,file_path):
		self.file_path = file_path
		self.tokenizer = nltk.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
		self.lancaster = nltk.stem.lancaster.LancasterStemmer()

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
		self.logger = logging.getLogger(__name__)
	# creates vocabulary of the corpus

	def create_vocab(self):
		self.logger.info("Creating vocabulary")

		vocab = set()
		for i in self.instances:
			tokens = self.tokenizer.tokenize(i[0])
			tokens = [x for x in tokens if x not in nltk.corpus.stopwords.words('english')]
			tokens = [self.lancaster.stem(i) for i in tokens]
			vocab |= set(tokens)

		self.logger.debug("Vocabulary size: " + str(len(vocab)))
		return list(vocab)

	# extracts bag of words

	def bag_of_words(self):
		self.logger.info("Extacting bag of words features")
		X = np.empty([self.instances.shape[0],len(self.vocab)], dtype=float)
		text = ''
		output_file = self.file_path.split('.')[0] + ".features.txt"
		f = open(output_file,'w')
		for sentence in enumerate(self.instances):	
			tokens = self.tokenizer.tokenize(sentence[1][0])
			vector_set = {}
			vector_list = [0] * len(self.vocab)
			tokens = [x for x in tokens if x not in nltk.corpus.stopwords.words('english')]
			tokens = [self.lancaster.stem(i) for i in tokens]
			for t in tokens:
				index = self.vocab.index(t)
				if index in vector_set:
					vector_set[index] += 1
				else:
					vector_set[index] = 1
				vector_list[index] += 1
			X[sentence[0]] = vector_list
			text += str(self.y[sentence[0]]) + "\t" + str(vector_set) +"\n"

		f.write(text.strip())
		f.close()
		self.logger.info("Feautres are written in" + str(output_file))

		X = np.append( np.ones((X.shape[0], 1)), X, axis=1)
		return X

	def initialize(self):
		self.logger.info("Reading" + str(self.file_path))

		with open(self.file_path) as file:
			if(self.file_path.split('.')[1] == "csv"):
				file_content = np.array([line.strip().decode('utf-8').split(',') for line in file])
			elif (self.file_path.split('.')[1] == "tsv"):
				file_content = np.array([line.strip().decode('utf-8').split('\t') for line in file])


		classes = file_content[:,0]
		self.y = (classes == 'positive') * 1
		self.instances = file_content[:,1:]

		self.logger.debug("Number of training instances: " + str(self.instances.shape[0]))
		self.logger.debug("Number of classes: " + str(len(set(self.y.tolist()))))

		self.vocab = self.create_vocab()
		self.X = self.bag_of_words()

	def get_X(self):
		return self.X

	def get_y(self):
		return self.y


			