import pandas as pd 
import re
#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


#1-Reading the data in
#Select just the feature needed to explore more
# We just keep the polarity and the tweet colomns

input_dataset = pd.read_csv("training.1600000.processed.noemoticon.csv",
				header=1, names =["polarity","col1","col2","col3","col4","tweet"],
				usecols =["polarity","tweet"]) 

#Change the polarity to have more visibility 
#0 -> neg
#1 -> pos
input_dataset['polarity'] = input_dataset['polarity'].replace(4,1)

# 2-Prepare the dataset

# Function to Clean the tweets
def clean_data(text) :
    
    text = str(text)
    
    #Convert text to lowercase
    text = text.lower()
    
    #remove mentions '@'
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    
    #remove URLs links
    text = re.sub('https?\S+','',text)
    
    #remove control strings
    text = re.sub(r'[\n\r\t]','',text)   
    
    #remove numbers
    text = re.sub(r'[0-9]+','',text)
    
    #remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    #remove stop words
    
    #remove the added spaces
    text = text.split()
    text = " ".join(text)
    
    return text

#apply the cleaning function to the tweets
input_dataset.tweet = input_dataset.tweet.apply(lambda x: clean_data(x))
print(input_dataset.tail())


#3-Start Modeling
#We will work with the logistic regression

#First, we put tweet and polarity in vectors
X_train = input_dataset['tweet']
Y_train = input_dataset['polarity']


#Then, we vectorize considering unigrams, bigrams and trigrams
#(more efficient to reconize for exp that "not happy" is negative unstead of consedring just happy and return a false result)
vectoriser = TfidfVectorizer(ngram_range=(1,3), max_features = 500000)
vectoriser.fit(X_train)

# Term Frequency, Inverse Document Frequency measures the importance of each word in the message and compare it against all the messages to detemine its importance
X_train = vectoriser.transform(X_train)


### Finally we create our logisticRegression models
model = LogisticRegression()
model.fit(X_train, Y_train)


print("end of modeling")

