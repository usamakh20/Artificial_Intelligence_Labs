import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords


train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3) 

mylist=[]

def filterData(trainData):
    for i in trainData['review']:
        e1 = BeautifulSoup(i,"html.parser")#removes html tags
        e1=e1.get_text()
        e1=re.sub('[^a-zA-Z]',' ',e1)#removes special characters
        e1=e1.lower().split()#converts to lower case and splits data 
        stops = set(stopwords.words('english'))#identifies stop words
        e1 = [w for w in e1 if not w in stops]#removes stopwords
        e1=" ".join(e1)#joins the words back with a space
        mylist.append(e1)
    return mylist

#print(filterData(train[:2]))

"""
sentiment= train["sentiment"][:10]
Y = [1,1,1,0,0,0,0,0]    
print Y
myylist=[]

print sentiment.tolist()
"""
