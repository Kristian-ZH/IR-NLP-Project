from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

sys.path.insert(1, 'utils')
sys.path.insert(1, 'DataGeneration')
sys.path.insert(1, 'VectorSpaceModel')

import utils
from BinaryIndependenceModel.BIMQuery import BIMQuery
import VectorSpaceModel

data = utils.loadDate('DataGeneration/people', 22)

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
        preprocessed_query = utils.preprocessText(query)

        result = VectorSpaceModel.vectorSpaceModel(data, vec, corpus, preprocessed_query, 5)
        return {'result': appendUrl(result)}, 200  # return data and 200 OK

class BIM(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('query', required=True)  # add args
        args = parser.parse_args()
        query = args['query']

        # query="Acted in both Breadking Bad and it's spinoff Better Call Saul"
        preprocessed_query = utils.preprocessText(query)

        result = BIMQuery(data, preprocessed_query, 5)
        print(appendUrl(result))
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
