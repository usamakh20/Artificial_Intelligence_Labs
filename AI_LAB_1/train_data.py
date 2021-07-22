from __future__ import division
import clean_data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",max_features = 3000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_size= int(0.8*len(data["review"]))
reviews = clean_data.clean(data["review"])
labels = list(data["sentiment"])
X = vectorizer.fit_transform(reviews[:train_size]).toarray()

# Numpy arrays are easy to work with, so convert the result to an
# array
Y=labels[:train_size]

clf = MultinomialNB(alpha=5) # alpha=0 means no laplace smoothing
clf.fit(X, Y)


# bag of word representation
tX = vectorizer.transform(reviews[train_size:]).toarray()
# prediction
predicted= clf.predict(tX)
actual = labels[train_size:]
similar =0

for i in xrange(len(predicted)):
    if int(actual[i])==int(predicted[i]):
        similar+=1

print "Prediction accuracy is: "+str(round(similar/len(predicted)*100,2))+"%"
