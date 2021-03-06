import math

def dft(documents):
    res = [0] * 65536
    for doc in documents:
        for i in range(len(doc["vec"])):
            if doc["vec"][i] == 1:
                res[i] += 1
    return res


def BIM(documents, query, firstN):
    res = []
    df = dft(documents)
    for doc in documents:
        rsvd = 0
        for i in range(len(query)):
            if query[i] == 1 and doc["vec"][i] == 1:
                Ut = df[i] / len(documents)
                Pt = 0.33 + 0.66 * Ut
                rsvd += math.log(Pt / (1 - Pt)) + math.log((1.00001 - Ut) / Ut)
        res.append({"doc_name": doc["doc_name"], "score": rsvd})
    res.sort(key=lambda x: x["score"], reverse=True)
    return res[0:firstN]

dict = {}

def vectorize(doc):
    res = [0] * 65536
    for word in doc:
        if word in dict.keys():
            res[dict[word]] = 1
        else:
            dict[word] = len(dict)
            res[dict[word]] = 1
    return res

def BIMQuery(data, query, firstN):
    listObj = []
    for doc in data.keys():
        obj = {"doc_name": doc, "doc": data[doc]}
        wordArr = data[doc].split(" ")
        obj["vec"] = vectorize(wordArr)
        listObj.append(obj)
    queryvec = vectorize(query.split(" "))
    #print(dict)
    return list(map(lambda x : x["doc_name"], BIM(listObj, queryvec, firstN)))

