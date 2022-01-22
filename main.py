import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

sys.path.insert(1, 'DataGeneration')
sys.path.insert(1, 'VectorSpaceModel')

from BinaryIndependenceModel.BIMQuery import BIMQuery
import VectorSpaceModel

def loadDate(path): # data-generation/people
    files = os.listdir(path)
    data = pd.Series([],dtype=pd.StringDtype()) 

    for file in files:
        f = open(os.path.join(path, file), encoding="utf8")
        data[f.name[22:]]= f.read()
        f.close
    
    return data

# Lemmatizing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import 	WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
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
data = loadDate('dataGeneration/people')

preprocessData(data)

# VECTOR SPACE MODEL
# vec = TfidfVectorizer(norm=None, ngram_range=(1, 3))
vec = TfidfVectorizer(norm=None)
corpus = vec.fit_transform(data)




# query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
# preprocessed_query = preprocessText(query)

# print(VectorSpaceModel.vectorSpaceModel(data, vec, corpus, preprocessed_query, 5))
# print(BIMQuery(data, preprocessed_query, 5))



# Server
from flask import Flask, render_template, Response
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
import os.path


app = Flask(__name__)
api = Api(app)

def appendUrl(result):
    resultWithUrls = [{"name": person, "url": "https://en.wikipedia.org/wiki/" + person} for person in result]
    return resultWithUrls

# Logic
class VSM(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('query', required=True)  # add args
        args = parser.parse_args()
        query = args['query']

        # query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
        preprocessed_query = preprocessText(query)

        result = VectorSpaceModel.vectorSpaceModel(data, vec, corpus, preprocessed_query, 5)
        return {'result': appendUrl(result)}, 200  # return data and 200 OK

class BIM(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('query', required=True)  # add args
        args = parser.parse_args()
        query = args['query']

        # query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
        preprocessed_query = preprocessText(query)

        result = BIMQuery(data, preprocessed_query, 5)
        print("KRIS")
        print(appendUrl(result))
        print("KRIS")
        return {'result': appendUrl(result)}, 200  # return data and 200 OK

# class Boolean(Resource):
#     def get(self):
#         parser = reqparse.RequestParser()  # initialize
#         parser.add_argument('query', required=True)  # add args
#         args = parser.parse_args()
#         query = args['query']

#         # query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
#         preprocessed_query = preprocessText(query)

#         result = VectorSpaceModel.vectorSpaceModel(data, vec, corpus, preprocessed_query, 5)
#         return {'result': appendUrl(result)}, 200  # return data and 200 OK

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)

@app.route('/')
def index_page():
    content = get_file('frontend/index.html')
    return Response(content, mimetype="text/html")

api.add_resource(VSM, '/vsm')  # add endpoints
api.add_resource(BIM, '/bim')  # add endpoints
#api.add_resource(BIM, '/boolean')  # add endpoints

if __name__ == '__main__':
    app.run()
