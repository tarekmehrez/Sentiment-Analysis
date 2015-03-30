# Sentiment-Analysis
Implementation of logistic regressio to perform sentiment analysis

The tool takes in csv/tsv annotated text files with 2 (+ve,-ve) classes at the beginning of each instance,
then it outputs the classification accuracy based on the parameters entered by the user

# Usage

To run the classifier: `python main.py [options]`

```

usage: main.py [-h] [--file FILE] [--train {0,1}] [--iter ITER]
               [--alpha ALPHA] [--perc PERC] [--reg REG] [--feat {0,1}]
               [--vec-size VEC]

  -h, --help      show this help message and exit
  --file FILE     Input training file (csv or tsv) with class at the beginning
                  [default=corpus/movie-reviews-sentiment.tsv]
  --train {0,1}   Training algorithm: 0 for logistic regression & 1 for RNTN
                  [default=0]
  --iter ITER     Number of iterations [default=1000]
  --alpha ALPHA   Learning rate [default=0.0005]
  --perc PERC     Percentage split for training & testing data [default=0.8]
  --reg REG       Regularization Paramter [default=100]
  --feat {0,1}    Type of features 0: bag of words, 1: word2vec [default=0]
  --vec-size VEC  Vector size in case --feat was set to 1 (word2vec), if bag-
                  of-words is used, vec-size is ignored [default=2000]

```
