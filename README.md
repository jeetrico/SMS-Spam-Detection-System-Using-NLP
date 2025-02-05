# SMS Spam Detection System Using NLP (P1) - A Project by Jeet Banerjee

## Overview

This project implements a machine learning system for detecting spam SMS messages. It uses Natural Language Processing (NLP) techniques to classify messages as either "spam" (unwanted advertisements, malicious content) or "ham" (legitimate messages).

## Technologies Used

*   Python
*   Pandas
*   Scikit-learn
*   NLTK (Natural Language Toolkit)
*   Matplotlib
*   Seaborn
*   `joblib` (for model persistence)
*  `imbalanced-learn` (to use for balanced datasets)
These have been included and a summary of the purpose is in this file in case they would like to know how to add them using pip.

## Project Stages

This project involves the following key steps:

*   **Data Collection:** Gathering a dataset of SMS messages labeled as spam or ham.
*   **Data Cleaning and Preprocessing:** Cleaning and preparing the text data for analysis.
*   **Exploratory Data Analysis (EDA):** Gaining insights into the data through visualizations and statistical analysis.
*   **Feature Engineering:** Creating new features from the text data to improve model performance.
*   **Model Building and Selection:** Training and evaluating different machine learning models.
*   **Model Persistence:** Saving the trained model and vectorizer for later use.
*   **Testing the process:** This phase provides different ways to test this, ranging from uploading different text/data on websites or uploading all necessary codes for personal use.

## Data Collection

The SMS Spam Collection dataset was used to build and evaluate this project. The dataset contains a collection of SMS messages labeled as either "spam" or "ham".

## Data Cleaning and Preprocessing

The cleaning and preprocessing steps performed on the dataset were :
*   It handles encoding errors (UTF-8, Latin-1)
*   Removal of extra columns/re-naming to improve readability
*   It removes all duplicates and null or NAN values.

This cleans up the dataset so future models have more success with these features.
## Feature Engineering

The following features are engineered from each SMS message:

*   **Text Statistics:** The number of characters, words, and sentences in the message.
*   **TF-IDF Vectors:** Term Frequency-Inverse Document Frequency (TF-IDF) vectors represent the importance of words in the message relative to the entire dataset.

## Model Building and Selection

The code uses and tests the dataset on MultinomialNB, a variant of Naive Bayes that is proven to give good tests

**TF-IDF Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to reflect how important a word is to a document in a collection or corpus. The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general

*   **I prioritized high precision because** it minimizes the risk of incorrectly classifying legitimate messages as spam, which is crucial for ensuring a positive user experience.
*   **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance and prevent the model from being biased towards the majority class.

## Model Performance (Multinomial Naive Bayes)

*   **Accuracy:** 97% (Indication of the model being correct)
*   **Precision:** 83% (A more higher rate allows there to have a real correct test. 
*   **Used testing and strat test** To minimize the error between the two types of source data.
   
## Using the saved model
*For anyone you want to show it, it requires you to have a new project set so it can install*

1.  Open new Code on Colab.
2.  Import
3.    \```python
from joblib import load
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
#Importing Library to use
ps = PorterStemmer()
#Loading dependecies, should already be done here

def transform_text(text):
    """Converts text to lowercase, tokenizes, removes special characters, stopwords, and punctuation, and applies stemming."""
    if not isinstance(text, str):
        return ""  # Handle non-string input
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    alphanumeric_tokens = [token for token in tokens if token.isalnum()]
    filtered_tokens = [token for token in alphanumeric_tokens if token not in stopwords.words('english') and token not in string.punctuation]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)

try:
    mnb = load('spam_model.joblib') #load the model
    tfidf = load('tfidf_vectorizer.joblib') #load the encoder
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model or vectorizer: {e}. Make sure the files are in the same directory as the script.")
    exit() #stop program

while True:
    sms_input = input("Enter an SMS message (or type 'exit' to quit): ")
    if sms_input.lower() == 'exit':
        break
    processed_text = transform_text(sms_input)
    vectorized_text = tfidf.transform([processed_text])
    prediction = mnb.predict(vectorized_text)[0]

    if prediction == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM")

  \```
