import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

sys.path.insert(1, 'DataGeneration')
sys.path.insert(1, 'VectorSpaceModel')
import GeneratePeople
import VectorSpaceModel

def loadDate(path): # data-generation/people
    files = os.listdir(path)
    data = pd.Series([],dtype=pd.StringDtype())

    for file in files:
        f = open(os.path.join(path, file),'r')
        data[f.name[22:]]= f.read()
        f.close
    
    return data

# Lemmatizing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import 	WordNetLemmatizer

nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocessText(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    filtered_tokens = [w for w in word_tokens if not w.lower() in stop_words]

    for filtered_idx, w in enumerate(filtered_tokens):
            filtered_tokens[filtered_idx] = wordnet_lemmatizer.lemmatize(w.lower())

    return TreebankWordDetokenizer().detokenize(filtered_tokens)

def preprocessData(data):
    for idx, text in enumerate(data):
        data[idx] = preprocessText(text)

# Flow
#GeneratePeople.generateTexts('./corpus/combined.txt')
data = loadDate('dataGeneration/people')
preprocessData(data)

# VECTOR SPACE MODEL
# vec = TfidfVectorizer(norm=None, ngram_range=(1, 3))
vec = TfidfVectorizer(norm=None)
corpus = vec.fit_transform(data)


# LOOP
query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
preprocessed_query = preprocessText(query)
print(VectorSpaceModel.vectorSpaceModel(data, vec, corpus, preprocessed_query, 5))