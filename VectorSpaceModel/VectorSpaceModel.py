import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def getFirstNMatches(data, corpus, query, firstNMatches):
    sims = cosine_similarity(corpus, query)
    sims_tuples = [(data.index.tolist()[idx],sim) for idx, sim in enumerate(sims)]
    sims_tuples.sort(key=lambda x:x[1], reverse=True)
    result = [match[0] for match in sims_tuples[:firstNMatches]]

    return result

def vectorSpaceModel(data, vec, corpus, query, firstNMatches):
    tf_idf_query = vec.transform([query])
    firstNMatches=getFirstNMatches(data, corpus, tf_idf_query, firstNMatches)

    return firstNMatches
