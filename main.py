from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy import displacy
from spacy.matcher import PhraseMatcher
from pathlib import Path
import os
import sys

sys.path.insert(1, 'utils')
sys.path.insert(1, 'DataGeneration')
sys.path.insert(1, 'VectorSpaceModel')
sys.path.insert(1, 'BinaryIndependenceModel')
sys.path.insert(1, 'Boolean')

import utils
import VectorSpaceModel
import BinaryIndependenceModel
import Boolean


colorMap = {
    "ADJ": "blue",
    "ADP": "cyan",
    "ADV": "orange",
    "AUX": "green",
    "CONJ": "brown",
    "CCONJ": "brown",
    "DET": "magenta",
    "INTJ": "yellow",
    "NOUN": "red",
    "PART": "azure",
    "PRON": "pink",
    "PROPN": "purple",
    "SCONJ": "brown",
    "VERB": "green",
    "X": "gray",
}

data = utils.loadDate('DataGeneration/people', 22)

boolSearch,doc_ids = Boolean.prepare("DataGeneration/people")

# Lemmatizing
utils.preprocessData(data)

# VECTOR SPACE MODEL
# vec = TfidfVectorizer(norm=None, ngram_range=(1, 3))
vec = TfidfVectorizer(norm=None)
corpus = vec.fit_transform(data)


# query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
# preprocessed_query = preprocessText(query)

# print(VectorSpaceModel.vectorSpaceModel(data, vec, corpus, preprocessed_query, 5))
# print(BIMQuery(data, preprocessed_query, 5))


# Server
from flask import Flask, Response
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

def fetchKeysentences(person, query):
        f = open(os.path.join('DataGeneration/people', person), encoding="utf8")
        text = f.read()
        f.close

        nlp = spacy.load("en_core_web_sm")
        phrase_matcher = PhraseMatcher(nlp.vocab)
        phrases = query.split(' ')
        patterns = [nlp(text) for text in phrases]
        phrase_matcher.add('person', None, *patterns)

        doc = nlp(text)
        sentences = ""
        added = []
        for sent in doc.sents:
            if len(added) == 6:
                    break
            for match_id, start, end in phrase_matcher(nlp(sent.text)):
                if nlp.vocab.strings[match_id] in ["person"] and sent.text not in added:
                    sentences += sent.text.replace("&", "and")
                    added.append(sent.text)

        return sentences

def formatResults(query, result):
    resultWithUrls = [{"name": person.replace("_", " "), "url": "https://en.wikipedia.org/wiki/" + person, "summary": fetchKeysentences(person, query)} for person in result]
    return resultWithUrls

# Logic
class VSM(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('query', required=True)  # add args
        args = parser.parse_args()
        query = args['query']

        # query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
        preprocessed_query = utils.preprocessText(query)

        result = VectorSpaceModel.vectorSpaceModel(data, vec, corpus, preprocessed_query, 5)

        return {'result': formatResults(query, result)}, 200  # return data and 200 OK

class BIM(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('query', required=True)  # add args
        args = parser.parse_args()
        query = args['query']
        print(query)
        # query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
        preprocessed_query = utils.preprocessText(query)

        result = BinaryIndependenceModel.BIMQuery(data, preprocessed_query, 5)
      
        return {'result': formatResults(query, result)}, 200  # return data and 200 OK

class Bool(Resource):
     def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('query', required=True)  # add args
        args = parser.parse_args()
        query = args['query']

        # query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
        preprocessed_query = query

        result = Boolean.search(doc_ids,boolSearch,preprocessed_query,5)
        return {'result': formatResults(query,result)}, 200  # return data and 200 OK

class Color(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('sentences', required=True)  # add args
        args = parser.parse_args()
        sentences = args['sentences']

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentences)
        taggedWords= []

        for token in doc:
            taggedWords.append({'word': token.text,  'color':colorMap[token.pos_] if token.pos_ in colorMap else 'black'})

        return taggedWords

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
api.add_resource(Bool, '/boolean')  # add endpoints
api.add_resource(Color, '/color')  # add endpoints

if __name__ == '__main__':
    app.run()
