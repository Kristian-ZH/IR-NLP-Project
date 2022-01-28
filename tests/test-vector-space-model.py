import json
from os import walk
from sklearn.metrics import recall_score
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(1, '../VectorSpaceModel')
sys.path.insert(1, '../utils')

import VectorSpaceModel
import utils

def loadQueries():
    f = open('queries.json')
    data = json.load(f)
    queries_result = [(query['query'], query['result']) for query in data['queries']]
    f.close()

    return queries_result

expected_result = loadQueries()

data = utils.loadDate('../DataGeneration/people', 25)
utils.preprocessData(data)
vec = TfidfVectorizer(norm=None)
corpus = vec.fit_transform(data)

query_result = [VectorSpaceModel.vectorSpaceModel(data, vec, corpus, utils.preprocessText(query[0]), 10) for query in expected_result]

def to_vector(docs, doc_set):
    return [1 if doc in doc_set else 0 for doc in docs]

def testVectorSpaceModel():
    filenames = next(walk('../DataGeneration/people'), (None, None, []))[2]

    gold_v = [to_vector(filenames, _v[1]) for _v in expected_result]
    predicted_v = [to_vector(filenames, _v) for _v in query_result]

    overall_recall=0
    queries_sum=0
    for i in range(len(gold_v)):
        overall_recall += recall_score(gold_v[i], predicted_v[i])
        queries_sum+=1
    
    return overall_recall/queries_sum

recall_score = testVectorSpaceModel()
print(recall_score)