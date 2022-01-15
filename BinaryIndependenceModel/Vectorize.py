import glob
import re
import nltk
from BIM import BIM
nltk.download('stopwords')
from nltk.corpus import stopwords

dict = {}


def vectorize(doc):
    res = [0] * 32768
    doc = doc.lower()
    doc = tokenizer.tokenize(doc)
    for word in doc:
        if word in dict.keys():
            res[dict[word]] = 1
        else:
            dict[word] = len(dict)
            res[dict[word]] = 1
    return res

list = []
files = glob.glob("people/*.txt")
for file in files:
    opened = open(file, encoding="utf8")
    new_file = opened.read().replace("\n", " ").replace("=", " ").lower()
    new_file = re.sub(' +', ' ', new_file)
    obj = {"doc_name": file, "doc": new_file}
    list.append(obj)


tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words('english'))
for doc in list:
    doc["vec"] = vectorize(doc["doc"])


def BIMQuery(query):
    queryvec = vectorize(query)
    print(BIM(list, queryvec))

BIMQuery("Is a producer and director")
