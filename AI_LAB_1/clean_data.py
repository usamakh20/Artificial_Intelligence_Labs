from bs4 import BeautifulSoup
import re
import nltk

stops = set(nltk.corpus.stopwords.words('english'))

def review_to_words(raw_review):
    raw_review = BeautifulSoup(raw_review, "html.parser").get_text()
    raw_review = re.sub('[^a-zA-Z]', " ", raw_review)
    review = raw_review.lower().split()
    review = [w for w in review if not w in stops]
    return " ".join(review)

def clean(train):
    processed_review=[]
    # Run the BeautifulSoup object on every single movie review
    for i in xrange(0,len(train)):
        processed_review.append(review_to_words(train[i]))
    return processed_review

