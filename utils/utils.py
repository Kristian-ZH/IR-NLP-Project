import nltk
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import 	WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))


def loadDate(path, nameOffset): # data-generation/people
    files = os.listdir(path)
    data = pd.Series([],dtype=pd.StringDtype()) 

    for file in files:
        f = open(os.path.join(path, file), encoding="utf8")
        data[f.name[nameOffset:]]= f.read()
        f.close
    
    return data

def preprocessText(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    filtered_tokens = [w for w in word_tokens if not w.lower() in stop_words]

    for filtered_idx, w in enumerate(filtered_tokens):
        if w == "&":
            w = "and"
        filtered_tokens[filtered_idx] = wordnet_lemmatizer.lemmatize(w.lower())

    return TreebankWordDetokenizer().detokenize(filtered_tokens)

def preprocessData(data):
    for idx, text in enumerate(data):
        data[idx] = preprocessText(text)