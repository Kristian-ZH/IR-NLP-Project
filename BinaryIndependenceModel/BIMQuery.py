from BinaryIndependenceModel.BIM import BIM

dict = {}
listObj = []

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
    for doc in data.keys():
        obj = {"doc_name": doc, "doc": data[doc]}
        wordArr = data[doc].split(" ")
        obj["vec"] = vectorize(wordArr)
        listObj.append(obj)
    queryvec = vectorize(query.split(" "))
    #print(dict)
    return list(map(lambda x : x["doc_name"], BIM(listObj, queryvec, firstN)))
