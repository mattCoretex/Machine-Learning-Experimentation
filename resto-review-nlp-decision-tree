#Natural Language Processing using Decision Tree classification
#train a model to predict if a restaurant review is positive or negative
#Author: Mek O.
#Email: aobchey00@uvic.ca
#lecture by SuperDataScience on Udemy


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Section 36 - Natural Language Processing/Restaurant_Reviews.tsv', sep = '\t', quoting = 3)

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#a collection of texts
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    #stem the word - keep on only the root
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    #if working with articles, set(stopwords.words('english')) makes the algorithm executes faster
    review = ' '.join(review)
    corpus.append(review)
    
#creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500) #max_features keep only most K relevant words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

#fitting kernel SVM classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)
#predicting the test set result
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Performance metrics
tp = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tn = cm[1,1]

acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * prec * recall / (prec + recall)
