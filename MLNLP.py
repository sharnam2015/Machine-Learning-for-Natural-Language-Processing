# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Code is developed for the IMDB Movies Dataset Kaggle Challange - A Natural Language Processing Challenge
#In this code Bag of words and random forest is used for NLP. The built model is
#tested on the remaining 5000 rows of the labeled data set and then applied to test data
#Author - Sharnam Shah Last Date Modified - 03/16/2020, some of the code is also taken from public online sources

import pandas as pd
import csv
import numpy as np
import re
import math
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup 
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  

#function to make the wordcloud
def show_wordcloud(data, title = None):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

if __name__ == '__main__':
    
    #Reading the entire labeled dataset and counting the positive and negative sentiment reviews
    trainoriginal = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
    print ("number of rows for sentiment 1 in whole dataset: {}".format(len(trainoriginal[trainoriginal.sentiment == 1])))
    print ( "number of rows for sentiment 0 in whole dataset: {}".format(len(trainoriginal[trainoriginal.sentiment == 0])))
    
    #using only the first 20000 rows of the labeled dataset to train the model
    train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3, nrows=20000)
        
    #computing the number of rows in the training dataset with positive and negative sentiment     
    print ("number of rows for sentiment 1 in the first 20000 rows: {}".format(len(train[train.sentiment == 1])))
    print ( "number of rows for sentiment 0 in the first 20000 rows: {}".format(len(train[train.sentiment == 0])))
    
    #computing the average string length of the reviews
    l=0
    length = [(l + len(x)) for x in train["review"]]
    avgstring_length = sum(length)/len(train["review"])
    print("The average string length is:",avgstring_length)
    
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size

    # Initialize an empty list to hold the clean reviews
    #Parsing training set movie reviews
    print ("Cleaning and parsing the training set movie reviews...\n")
    
    clean_train_reviews = []
    for i in range( 0, num_reviews ):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print ("Review %d of %d\n" % ( i+1, num_reviews ))                                                                    
        clean_train_reviews.append( review_to_words( train["review"][i] ))
            
    #creating a positive review set and then a wordcloud for the same
    clean_train_revpos = []
    for i in range( 0, num_reviews ):
         if (train.sentiment[i]==1):
             clean_train_revpos.append(review_to_words(train["review"][i]))
    print("Showing the wordcloud for positive sentiments")
    show_wordcloud(clean_train_revpos)
    
    #creating a negative review set and then a wordcloud for the same
    clean_train_revneg = []
    for i in range( 0, num_reviews ):
         if (train.sentiment[i]==0):
             clean_train_revneg.append(review_to_words(train["review"][i]))
    print("Showing the wordcloud for negative sentiments")
    show_wordcloud(clean_train_revneg)
           
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    train_data_features = train_data_features.toarray()
    vocab = vectorizer.get_feature_names()
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    print ("Training the random forest...")
    
    # Initialize a Random Forest classifier with 300 trees
    forest = RandomForestClassifier(n_estimators = 300) 
    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )
    
    #Reading the last 5000 rows of the labeled data and using it to get the accuracy of the model
    actus = pd.read_csv("labeledTrainData.tsv", header=0, \
                        delimiter = "\t",quoting=3,skiprows = 20000,nrows=5000,names=["id","sentiment","review"])
        
    # Create an empty list and append the clean reviews one by one
    num_reviews = len(actus["review"])
    clean_act_reviews = []
    
    print ("Cleaning and parsing the model trial 5000 label set movie reviews...\n")
    for i in range(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print ("Review %d of %d\n" % (i+1, num_reviews))
        clean_review = review_to_words( actus["review"][i] )
        clean_act_reviews.append( clean_review )

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_act_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    resultmodeltrial = forest.predict(test_data_features)
    
    #Getting the confusion matrix, accuracy of the model prediction for the last 5000 lines of labeled train data
    actual=actus.sentiment
    predicted=resultmodeltrial
    cm=confusion_matrix(actual,predicted)
    print("Built Model is compared to the labels for the last 5000 rows of labeled train dataset")
    print("Confusion Matrix is:",cm)
    tn, fp, fn, tp = cm.ravel()
    Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
    print("Model Accuracy is:" ,Accuracy)
    Precision = tp/(tp+fp) 
    print("Model Precision is:",Precision)
    
    #Now using this model on the test data
    # Read the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                       quoting=3 )

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = [] 

    print ("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print ("Review %d of %d\n" % (i+1, num_reviews))
        clean_review = review_to_words( test["review"][i] )
        clean_test_reviews.append( clean_review )

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
    
    
    