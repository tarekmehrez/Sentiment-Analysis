import argparse
import sys
import logging

from corpus import Corpus
from logistic_regression import LogisticRegression

def help_exit():
	parser.print_help()
	sys.exit(1)

parser = argparse.ArgumentParser()

parser.add_argument('--file', action='store', dest='file',
                    help='Input training file (csv or tsv) with class at the beginning')

parser.add_argument('--train', action='store', dest='train',type=int,default=0,choices=[0, 1],
                    help='Training algorithm: 0 for logistic regression & 1 for RNTN [default=0]')

parser.add_argument('--iter', action='store', dest='iter',type=int,default=1000,
                    help='Number of iterations [default=1000]')

parser.add_argument('--alpha', action='store', dest='alpha',type=int ,default=0.0005,
                    help='Learning rate [default=0.0005]')

parser.add_argument('--perc', action='store', dest='perc',type=int ,default=0.8,
                    help='Percentage split for training & testing data [default=0.8]')

parser.add_argument('--feat', action='store', dest='feat',type=int, default=0,choices=[0, 1, 2],
                    help='Type of features 0: bag of words, 1: glove, 2: word2vec [default=0]')


results = parser.parse_args()
if len(sys.argv)==1:
	help_exit()


if results.iter < 0:
	print "Expected iterations value greater than 0"
	help_exit()

if results.alpha < 0:
	print "Expected alpha value greater than 0"
	help_exit()

if results.perc < 0 or results.perc > 1:
	print "Expected Percentage value between 0 & 1"
	help_exit()


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

logger.debug('Started with options:'		+ "\n" +
			'file:	' + str(results.file) 	+ "\n" + 
			'train:	' + str(results.train)	+ "\n" + 
			'iter:	' + str(results.iter)	+ "\n" + 
			'alpha:	' + str(results.alpha) + "\n" + 
			'perc:	' + str(results.perc)	+ "\n" + 
			'feat:	' + str(results.feat))

corpus = Corpus(results.file)
corpus.initialize()

if results.train == 0:
	classifier = LogisticRegression(corpus.get_X(),corpus.get_y(),results.iter,results.alpha,results.perc)
	classifier.train()

