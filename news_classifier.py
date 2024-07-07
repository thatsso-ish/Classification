import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd 
import pickle
import re
import seaborn as sns
import streamlit as st

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tag import pos_tag
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

from spellchecker import SpellChecker

spell = SpellChecker()
stop_words = stopwords.words('english')

#load pickles objects
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
modelNB = pickle.load(open("MultinomialNB.pkl", "rb"))


def df_processor(dataframe):
    df = dataframe.copy()

    #df['all_text'] = df['headlines'].astype(str) + " " + df['description'].astype(str) + " " + df['content'].astype(str)
    #df.drop(['url', 'headlines', 'description', 'content'], axis=1, inplace=True)

    if df.isnull().values.any():
        print('Null values found in the dataset. Dropping rows with nulls...\n')
        df.dropna(inplace=True)
        print('Null values removed.\n')
    else:
        print('No null values in the dataset\n')

    prestring = df['all_text'].iloc[0]
    df['all_text'] = df['all_text'].str.lower()
    poststring = df['all_text'].iloc[0]

    print(f'\nStrings before converting to lowercase:\n{prestring}\n\nStrings after converting to lower:{poststring}')

    def clean_text(text):
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s!]', ' ', text)
        text = re.sub(r'\d+[-/]\d+[-/]\d+|(\d{1,3}(st|nd|rd|th)?)', ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        return text

    prestring = df['all_text'].iloc[0]
    df['all_text'] = df['all_text'].apply(clean_text)
    poststring = df['all_text'].iloc[0]

    print(f'\nStrings before cleaning:\n{prestring}\n\nStrings after cleaning:{poststring}')

    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    negation_words = {"not", "no", "never", "n't"}

    def tokenize_and_process(text):
        words = nltk.word_tokenize(text)
        words = [word for word in words if word.lower() not in stop_words]
        words = [lemmatizer.lemmatize(word, pos='v') for word in words]

        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, 10)
        bigrams = ["_".join(bigram) for bigram in bigrams]
        words.extend(bigrams)
        return " ".join(words)

    prestring = df['all_text'].iloc[0]
    df['all_text'] = df['all_text'].apply(tokenize_and_process)
    poststring = df['all_text'].iloc[0]

    print(f'\nStrings before tokenizing and removing stopwords:\n{prestring}\n\nStrings after tokenizing and removing stopwords:\n{poststring}')

    return df
    


def main():
    #collect user text
    news_text = st.text_area("Enter News Text: ", height = 200)

    #convert user text into a df
    userdf = pd.DataFrame({'all_text': [news_text]})

    #perform preprocessing using our preprocessing function - df_processor
    cleandf = df_processor(userdf)

    #convert to features using my vectorizer
    X_ft = vectorizer.transform(cleandf['all_text'])


    #perform a prediction on the vectorized text
    y_pred = modelNB.predict(X_ft)

    #inverse transform and print the category
    y_tran = le.inverse_transform(y_pred)
    readable_cat = y_tran[0].title()
    st.warning(cleandf.head())

if __name__ == "__main__":
    main()
