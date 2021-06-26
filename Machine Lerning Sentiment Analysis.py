import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#read the csv file and print the shape of the dataset
yelp = pd.read_csv("E:\MS SEM2\CIS 593\Project\Dataset\yelp.csv")
yelp.shape

#drop all the rows with NaN
yelp = yelp.dropna()
yelp.shape

#add one more column to find out the relation between length vs number of reviews vs stars
yelp['text_len'] = yelp['text'].apply(len)
g = sns.FacetGrid(data=yelp, col='stars')
g.map(plt.hist, 'text_len', bins=50)

#This shuffles the data which help us to get the random tuple each time
from sklearn.utils import shuffle
yelp = shuffle(yelp)

#Defined two list for getting the postitives and negative reviews
positive_reviews = []
negative_reviews = []

#Get the 1600 positive reviews
p_counter = 0
index = 0
while p_counter != 1600:
    st = yelp['stars'][index]
    if (st == 5):
        positive_reviews.append(yelp['text'][index])
        p_counter = p_counter + 1
    index = index + 1
print(len(positive_reviews))

#Get the 1600 negative reviews
n_counter = 0
index = 0
while n_counter != 1600:
    st = yelp['stars'][index]
    if(st == 1 or st == 2):
        negative_reviews.append(yelp['text'][index])
        n_counter = n_counter + 1
    index = index + 1
print(len(negative_reviews))
  
#Tokenizer which tokenise the text into works and clean the data to prepare for NLP
def my_tokenizer(text):
    text = text.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(text) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful (punctuation)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords.words('english')] # remove stopwords    
    return tokens

#Dictionary and list defination for further use
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
orig_reviews = []

#Create dictionary for word count --> word_count_map from positive and negative reviews
for review in positive_reviews:
    orig_reviews.append(review)
    tokens = my_tokenizer(review)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    orig_reviews.append(review)
    tokens = my_tokenizer(review)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

print("len(word_index_map):", len(word_index_map))


# Creation our input matrices by converting tokens to vector
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

#train and test feature selection
X = data[:,:-1]
Y = data[:,-1]

#Apply machine learning algrithms for data mining
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

#Classification using Logistic Regrassion
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='multinomial', tol=1e-2, solver='newton-cg', max_iter=15)
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
print("\nClassification Algotithm - LOGISTIC REGRESSION")
print("Accuracy     - ", accuracy_score(y_test, y_predict))
print("Precesion    - ", precision_score(y_test, y_predict, average='weighted'))

#Classification using SUPPORT VECTOR MACHINE
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn import svm
clf = svm.SVC(gamma=2, C=2)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("\nClassification Algotithm - SVC")
print("Accuracy     - ", accuracy_score(y_test, y_predict))
print("Precesion    - ", precision_score(y_test, y_predict, average='weighted'))

#Classification using DECISION TREE
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn import tree
dt = tree.DecisionTreeClassifier(max_depth=25)
dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)
print("\nClassification Algotithm - DECISION TREE")
print("Accuracy     - ", accuracy_score(y_test, y_predict))
print("Precesion    - ", precision_score(y_test, y_predict, average='weighted'))

#Classification using NAIVE BAYES
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_predict = nb.predict(X_test)
print("\nClassification Algotithm - NAIVE BAYES")
print("Accuracy     - ", accuracy_score(y_test, y_predict))
print("Precesion    - ", precision_score(y_test, y_predict, average='weighted'))







def tokens_to_vector_predict(tokens):
    x = np.zeros(len(word_index_map)) # last element is for the label
    for t in tokens:
        print(t)
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    return x

def predict(text):
    tokens = my_tokenizer(text)
    print(tokens)
    N = 1
    data = np.zeros((N, len(word_index_map)))
    data = tokens_to_vector_predict(tokens)

    y_predict = nb.predict(data)
    print("Predicted by Naive Bayes - ", y_predict)
    y_predict = lr.predict(data)
    print("Predicted by Logistic Regression - ", y_predict)
    y_predict = clf.predict(data)
    print("Predicted by SVM - ", y_predict)
    y_predict = dt.predict(data)
    print("Predicted by Decision Tree - ", y_predict)
    
#Predicting a singular review

review = yelp['text'][5012]
starCount = yelp['stars'][5012]
print("Original Review - \n", review)
print("Star count - ",starCount)
predict(review)




