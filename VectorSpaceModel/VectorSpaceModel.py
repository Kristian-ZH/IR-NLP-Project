import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def loadVSMData():
    path = 'data-generation/people'
    files = os.listdir(path)
    data = pd.Series([],dtype=pd.StringDtype())

    for file in files:
        f = open(os.path.join(path, file),'r')
        data[f.name[7:]]= f.read()
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
    for w in word_tokens:
        if w not in stop_words:
            filtered_tokens.append(w)
    for filtered_idx, w in enumerate(filtered_tokens):
            filtered_tokens[filtered_idx] = wordnet_lemmatizer.lemmatize(w.lower())

    return TreebankWordDetokenizer().detokenize(filtered_tokens)

def preprocessData(data):
    for idx, text in enumerate(data):
        data[idx] = preprocessText(text)

def getFirstNMatches(data, corpus, query, firstNMatches):
    sims = cosine_similarity(corpus, query)
    sims_tuples = [(data.index.tolist()[idx],sim) for idx, sim in enumerate(sims)]
    sims_tuples.sort(key=lambda x:x[1], reverse=True)
    result = [match[0][16:] for match in sims_tuples[:firstNMatches]]

    return result

def vectorSpaceModel(vec, corpus, query, firstNMatches):
    preprocessed_query = preprocessText(query)
    tf_idf_query = vec.transform([preprocessed_query])
    firstNMatches=getFirstNMatches(data, corpus, tf_idf_query, firstNMatches)

    return firstNMatches

data = loadVSMData()
preprocessData(data)

# vec = TfidfVectorizer(norm=None, ngram_range=(1, 3))
vec = TfidfVectorizer(norm=None)
corpus = vec.fit_transform(data)

# print(vectorSpaceModel(vec, corpus, "footballer France Forward", 3))