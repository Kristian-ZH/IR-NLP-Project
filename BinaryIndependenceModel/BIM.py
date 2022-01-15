import math


def dft(documents):
    res = [0] * 32768
    for doc in documents:
        for i in range(len(doc["vec"])):
            if doc["vec"][i] == 1:
                res[i] += 1
    return res


def BIM(documents, query):
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
    return res


#test = [{"doc_name": "asd1", "vec": [0, 1, 0, 0, 1, 3]},
#        {"doc_name": "asd2", "vec": [0, 0, 1, 0, 1, 0]},
#        {"doc_name": "asd3", "vec": [1, 0, 0, 0, 0, 0]},
#        {"doc_name": "asd4", "vec": [1, 1, 0, 0, 1, 3]},
#        {"doc_name": "asd5", "vec": [0, 1, 0, 0, 1, 3]},
#        {"doc_name": "asd6", "vec": [0, 0, 0, 1, 1, 0]}]
#BIM(test, [1, 1, 0, 0, 0, 0])
