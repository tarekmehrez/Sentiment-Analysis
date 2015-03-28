import numpy as np
import nltk
import logging
import gensim

class Corpus(object):

	def __init__(self,file_path,feat_type,vec):
		self.file_path = file_path
		self.feat_type = bool(feat_type)
		self.vec_size = vec


		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)


	# tokenize sentences, remove stop words & punctuation, and perform lancaster stemming

	def preprocess(self):
		self.logger.info("Preprocessing Corpus")
		tokenizer = nltk.RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
		lancaster = nltk.stem.lancaster.LancasterStemmer()
		self.sentences = []
		vocab = set()
		for i in self.instances:
			tokens = tokenizer.tokenize(i[0])
			tokens = [x for x in tokens if x not in nltk.corpus.stopwords.words('english')]
			tokens = [lancaster.stem(i) for i in tokens]
			self.sentences.append(tokens)
			vocab |= set(tokens)

		self.logger.debug("Vocabulary size: " + str(len(vocab)))
		return list(vocab)

	# extracts bag of words features

	def bag_of_words(self):
		self.logger.info("Extacting bag of words features")
		X = []
		for sentence in self.sentences:
			vector_list = [0] * len(self.vocab)
			for t in sentence:
				index = self.vocab.index(t)
				vector_list[index] += 1
			X.append(vector_list)

		X = np.asarray(X, dtype=float)
		X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
		return X

	# extract word vectors, each sentence is represented as average vector

	def word2vec(self):
		self.logger.info("Extracting features using Word2Vec")
		model = gensim.models.Word2Vec(self.sentences, min_count=0, window=2,size=self.vec_size)
		output_file = self.file_path.split('.')[0] + ".word2vec.txt"
		model.save(output_file)
		model.save_word2vec_format(output_file)

		with open(output_file) as file:
			next(file)
			file_content = np.array([line.strip().decode('utf-8').split() for line in file])

		vectors = dict()

		for i in file_content:
			x = i[1:].astype(float)
			vectors[i[0]] = x

		X = []
		for sentence in self.sentences:
			vector = np.zeros((self.vec_size,))
			for token in sentence:
				vector = vector + vectors[token]
			X.append(vector / float(len(sentence)))

		X = np.asarray(X, dtype=float)
		X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

		return X


	def initialize(self):
		self.logger.info("Reading " + str(self.file_path))

		with open(self.file_path) as file:
			if(self.file_path.split('.')[1] == "csv"):
				file_content = np.array([line.strip().decode('utf-8').split(',') for line in file])
			elif (self.file_path.split('.')[1] == "tsv"):
				file_content = np.array([line.strip().decode('utf-8').split('\t') for line in file])

		np.random.shuffle(file_content)
		np.random.shuffle(file_content)
		np.random.shuffle(file_content)

		classes = file_content[:,0]
		self.y = (classes == 'positive') * 1
		self.instances = file_content[:,1:]

		self.logger.debug("Number of instances: " + str(self.instances.shape[0]))
		self.logger.debug("Number of classes: " + str(len(set(self.y.tolist()))))

		if self.feat_type:
			self.vocab = self.preprocess()
			self.X = self.word2vec()
		else:
			self.vocab = self.preprocess()
			self.X = self.bag_of_words()

		self.logger.debug("Done extracting features with total feature space of size: " + str(self.X.shape))


	def get_X(self):
		return self.X

	def get_y(self):
		return self.y


			