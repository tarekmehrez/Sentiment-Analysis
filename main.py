import argparse
import sys
import logging

from corpus import Corpus
from logistic_regression import LogisticRegression

def help_exit():
	parser.print_help()
	sys.exit(1)

parser = argparse.ArgumentParser()

parser.add_argument('--file', action='store', dest='file', default='corpus/movie-reviews-sentiment.tsv',
                    help='Input training file (csv or tsv) with class at the beginning [default=corpus/movie-reviews-sentiment.tsv]')

parser.add_argument('--train', action='store', dest='train',type=int,default=0,choices=[0, 1],
                    help='Training algorithm: 0 for logistic regression & 1 for RNTN [default=0]')

parser.add_argument('--iter', action='store', dest='iter',type=int,default=1000,
                    help='Number of iterations [default=1000]')

parser.add_argument('--alpha', action='store', dest='alpha',type=float ,default=0.001,
                    help='Learning rate [default=0.0005]')

parser.add_argument('--perc', action='store', dest='perc',type=float ,default=0.9,
                    help='Percentage split for training & testing data [default=0.8]')

parser.add_argument('--reg', action='store', dest='reg',type=int ,default=100,
                    help='Regularization Paramter [default=100]')


parser.add_argument('--feat', action='store', dest='feat',type=int, default=0,choices=[0, 1],
                    help='Type of features 0: bag of words, 1: word2vec [default=0]')


parser.add_argument('--vec-size', action='store', dest='vec',type=int, default=2000,
                    help='Vector size in case --feat was set to 1 (word2vec), if bag-of-words is used, vec-size is ignored [default=2000]')


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

if results.reg < 0:
	print "Expected Regularization greater than 0"
	help_exit()

if results.vec < 0:
	print "Expected Vector Size greater than 0"
	help_exit()

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

logger.debug('Started with options:'		+ "\n" +
			'file:	' + str(results.file) 	+ "\n" + 
			'train:	' + str(results.train)	+ "\n" + 
			'iter:	' + str(results.iter)	+ "\n" + 
			'alpha:	' + str(results.alpha) 	+ "\n" +
			'reg:	' + str(results.reg) 	+ "\n" + 
			'perc:	' + str(results.perc)	+ "\n" + 
			'feat:	' + str(results.feat)	+ "\n" +
			'vec:	' + str(results.vec))

corpus = Corpus(results.file,results.feat,results.vec)
corpus.initialize()

if results.train == 0:
	classifier = LogisticRegression(corpus.get_X(),corpus.get_y(),results.iter,results.alpha,results.reg,results.perc)
	classifier.train()

