

from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from collections import defaultdict, Counter
from string import punctuation
import os
from os import scandir
from operator import itemgetter
from collections import defaultdict
from typing import Callable, Iterable
from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas, ParserElement,pyparsing_unicode, nums


tokenizer = TweetTokenizer()


# "C:\\Users\\Victor\\OneDrive\\AI\\IR\\IR-NLP-Project\\boolean\\test"

def tokenize_document(content):
    
    sentences = sent_tokenize(content)
    tokens = []
    for _sent in sentences:
        sent_tokens = tokenizer.tokenize(_sent)
        sent_tokens = [_tok.lower() for _tok in sent_tokens if _tok not in punctuation]
        tokens += sent_tokens
    
    return tokens

def tokenize_document_with_sentences(content):

    sentences = sent_tokenize(content)
    tokens = []
    for _sent in sentences:
        sent_tokens = tokenizer.tokenize(_sent)
        sent_tokens = [_tok.lower() for _tok in sent_tokens if _tok not in punctuation.replace(".","")]
        tokens += sent_tokens
    
    return tokens


def preprocess_document(content):
    return tokenize_document_with_sentences(content)


def prepare_dataset(documents_dir):

    tokenized_documents = []
    for document in os.listdir(documents_dir):
        with open(os.path.join(documents_dir, document), errors='ignore',encoding='utf8') as outf:
            tokenized_documents.append(preprocess_document(outf.read()))
    print("Found documents: ", len(tokenized_documents))
    
    return tokenized_documents 



def get_document_tokens(document_tokens,doc_id):
    #res = [(token, i) for token in document_tokens]
    res = []
    sentence_id = 0
    for position,token in enumerate(document_tokens):
        if token == ".":
            sentence_id=sentence_id+1
        else:
            res.append(((token,position,sentence_id),doc_id))

   # res=[((token,i),doc_id) for i,token in enumerate(document_tokens)]
    return res

def get_token_doc_id_pairs(category_dir):

    token_docid = []
    doc_ids = {}

    for i, document in enumerate(scandir(category_dir)):
        if document.is_file():
            doc_ids[i] = document.name
            with open(document,encoding='utf8') as out_fp:
                document_tokens = preprocess_document(out_fp.read())
                token_docid += get_document_tokens(document_tokens,i)
    return token_docid, doc_ids




# %%
def merge_token_in_doc(sorted_token_docid):
    """
    Returns a list of (token, doc_id, term_freq) tuples from a sorted list of (token, doc_id) list, 
    where if a token appears n times in a doc_id, we merge it in a tuple (toke, doc_id, n).
    """
    merged_tokens_in_doc = []
    for combined_token, doc_id in sorted_token_docid:
        (token,position,sentence) = combined_token
        if merged_tokens_in_doc:
            prev_tok, prev_doc_id, prev_freq,prev_positions,prev_sentences = merged_tokens_in_doc[-1]
            if prev_tok == token and prev_doc_id == doc_id:     
                merged_tokens_in_doc[-1] = (token, doc_id, prev_freq+1,prev_positions + [position],prev_sentences+[sentence])
            else:
                merged_tokens_in_doc.append((token, doc_id, 1,[position],[sentence]))
        else:
            merged_tokens_in_doc.append((token, doc_id, 1,[position],[sentence]))
    return merged_tokens_in_doc


def and_merge(list1,list2) -> list:
   # print(list1,list2)
    result = []
    ind1, ind2 = 0, 0
    while ind1 < len(list1) and ind2 < len(list2):
        el1, el2 = list1[ind1], list2[ind2]
        if el1 == el2:
            result.append(el1)
            ind1 += 1 
            ind2 += 1
        elif el1 < el2:
            ind1 += 1
        elif el1 > el2:
            ind2 += 1
    return result

def and_query(postings_word1, postings_word2):
    print(postings_word1)
    print(postings_word2)
    documents_results = []
    
    postings_ind1, postings_ind2 = 0, 0
    while postings_ind1 < len(postings_word1) and postings_ind2 < len(postings_word2):
        doc_id1, doc_id2 = postings_word1[postings_ind1][0], postings_word2[postings_ind2][0]
        if doc_id1 == doc_id2:
            documents_results.append((doc_id1,0))
            postings_ind1 += 1
            postings_ind2 += 1
        elif doc_id1 < doc_id2:
            postings_ind1 += 1
        elif doc_id1 > doc_id2:
            postings_ind2 += 1
    return documents_results

def sentence_query(postings_word1, postings_word2):
 
    documents_results = []
    
    postings_ind1, postings_ind2 = 0, 0
    while postings_ind1 < len(postings_word1) and postings_ind2 < len(postings_word2):
        doc_id1, doc_id2 = postings_word1[postings_ind1][0], postings_word2[postings_ind2][0]
        if doc_id1 == doc_id2:
            senList = and_merge(postings_word1[postings_ind1][3],postings_word2[postings_ind2][3])
            if(len(senList) > 0):
                documents_results.append((doc_id1,len(senList),[],senList))
            postings_ind1 += 1
            postings_ind2 += 1
        elif doc_id1 < doc_id2:
            postings_ind1 += 1
        elif doc_id1 > doc_id2:
            postings_ind2 += 1
    return documents_results

def sentence_multipar(lists) -> list:
    prev = lists[0]
    for i in range(0,len(lists)-1):
        prev = sentence_query(prev,lists[i+1])
    return prev

def and_multipar(self,lists) -> list:
    prev = lists[0]
    for i in range(0,len(lists)-1):
        prev = and_query(prev,lists[i+1])
    return prev

# %%
def or_query(postings_word1, postings_word2):

    documents_results = []
    
    postings_ind1, postings_ind2 = 0, 0
    while postings_ind1 < len(postings_word1) and postings_ind2 < len(postings_word2):
        doc_id1, doc_id2 = postings_word1[postings_ind1][0], postings_word2[postings_ind2][0]
        if doc_id1 == doc_id2:
            documents_results.append((doc_id1,0))
            postings_ind1 += 1
            postings_ind2 += 1
        elif doc_id1 < doc_id2:
            documents_results.append((doc_id1,0))
            postings_ind1 += 1
        elif doc_id1 > doc_id2:
            documents_results.append((doc_id2,0))
            postings_ind2 += 1
    if postings_ind1 == len(postings_word1):
        for i in range(postings_ind2,len(postings_word2)):
            documents_results.append((postings_word2[i][0],0))
    if postings_ind2 == len(postings_word2):
        for i in range(postings_ind1,len(postings_word1)):
            documents_results.append((postings_word1[i][0],0))
    return documents_results

def or_multipar(self,lists) -> list:
    prev = lists[0]
    for i in range(0,len(lists)-1):
        prev = or_query(prev,lists[i+1])
    return prev

# %%
def not_query(postings_word):
    document_count = len(tokenized_documents)
    documents_results = []

    prev = 0
    for i in range(0,len(postings_word)):
        for not_doc in range(prev,postings_word[i][0]):
            documents_results.append((not_doc,0))
        prev = postings_word[i][0]+1
    for not_doc in range(prev,document_count):
        documents_results.append((not_doc,0))
    
    return documents_results
        



class BoolRetrievalOperand:
    def __init__(self, t):
        self.label = t[0]
        print("Creating BoolRetrievalOperand" + str(t))
    
    def process(self) -> list:
        print("Processing BoolRetrievalOperand "+ self.label)
        self.value = postings[self.label]
        print(self.value)
        return self.value

    def __str__(self) -> str:
        return self.label

    __repr__ = __str__


# %%
class BoolRetrievalNot:
    def __init__(self, t):
        print("Creating BoolRetrievalNot"+str(t))
        self.arg = t[0][1]
        print(self.arg)


    def process(self) -> list:
        res = not_query(self.arg.process())
        print("Processing "+str(self))
        print(res)
        return res

    def __str__(self) -> str:
        return "~" + str(self.arg)

    __repr__ = __str__


class BoolRetrievalBinOp:
    repr_symbol: str = ""
    eval_fn: Callable[
        [list[list]], list
    ] = lambda _: []

    def __init__(self, t):
        print("Creating BoolRetrievalBinOp "+self.repr_symbol)
        print(t)
        self.args = t[0][0::2]

    def __str__(self) -> str:
        sep = " %s " % self.repr_symbol
        return "(" + sep.join(map(str, self.args)) + ")"


    def process(self) -> list:
        res = self.eval_fn([a.process() for a in self.args])
        print("Processing "+str(self))
        print(res)
        return res

    __repr__ = __str__


class BoolRetrievalAnd(BoolRetrievalBinOp):
    repr_symbol = "&"
    eval_fn = and_multipar


class BoolRetrievalOr(BoolRetrievalBinOp):
    repr_symbol = "|"
    eval_fn = or_multipar

def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

def get_wildcard_words(word):
    upperWord = word + "zzzzzzzzzzzzzzzzzzzzzzzzzz"
    keys = sorted(postings.keys())

   
    filtered1 = list(filter(lambda x: x >= word,keys))
    filtered2 = list(filter(lambda x: x >= upperWord,keys))
   
    print(Diff(filtered1,filtered2))
    return Diff(filtered1,filtered2)


# %%
def wildcard_process(word):
    wildcard_words = get_wildcard_words(word)
    print(wildcard_words[0])
    prew = postings[wildcard_words[0]]
    for i in range(1,len(wildcard_words)):
        prew = or_query(prew,postings[wildcard_words[i]])
    return prew

# %%

class BoolRetrievalWildcard:
    def __init__(self, t):
        self.label = t[0]
        print("Creating BoolRetrievalWildcard" + str(t))
    
    def process(self) -> list:
        print("Processing BoolRetrievalWildcard " + str(self))
        self.value = wildcard_process(self.label)
        print(self.value)
        return self.value

    def __str__(self) -> str:
        return self.label+"*"

    __repr__ = __str__

class BoolRetrievalProximity:
    def __init__(self, t):
        self.A = t[0]
        self.positions = t[2]
        self.B = t[3]
        print("Creating BoolRetrievalProximity" + str(t))
    
    def process(self) -> list:
        print("Processing BoolRetrievalProximity " + self.A + " \\" + self.positions + " " + self.B)
       # self.value = postings[ self.label]
      #  print(self.value)
        return process_proximity(self.A,self.B,self.positions)
        

    def __str__(self) -> str:
        return "" + self.A + " /" + self.positions + " " + self.B

    __repr__ = __str__

def process_sentence(words):
    converted = [postings[word] for word in words]
    print(converted)
    common_docs = sentence_multipar(converted)
    print(common_docs)
    return common_docs

def process_proximity(A,B,k):
    k = int(k)
    print(A,B,k)
    postings_word1,postings_word2 = postings[A],postings[B]
    documents_results = []
    
    postings_ind1, postings_ind2 = 0, 0
    while postings_ind1 < len(postings_word1) and postings_ind2 < len(postings_word2):
        doc_id1, doc_id2 = postings_word1[postings_ind1][0], postings_word2[postings_ind2][0]
        if doc_id1 == doc_id2:
            for k_ in range (k+1):
                
                print(list(map(lambda x: x + k_ + 1,postings_word1[postings_ind1][2])),postings_word2[postings_ind2][2])
                proxList = and_merge(list(map(lambda x: x + k_ + 1,postings_word1[postings_ind1][2])),postings_word2[postings_ind2][2])
                if(len(proxList) > 0):
                    documents_results.append((doc_id1,len(proxList),proxList,[]))
                    break
            postings_ind1 += 1
            postings_ind2 += 1
        elif doc_id1 < doc_id2:
            postings_ind1 += 1
        elif doc_id1 > doc_id2:
            postings_ind2 += 1

    print(documents_results)
    return documents_results

class BoolRetrievalSentence:
    def __init__(self, t):
        self.words = t[1:]
        print("Creating BoolRetrievalSentence" + str(t))
    
    def process(self) -> list:
        print("Processing BoolRetrievalSentence " + str(self.words))
       # self.value = postings[ self.label]

        print(self.words)
        return process_sentence(self.words)

    def __str__(self) -> str:
        return "/s r"+str(self.words)

    __repr__ = __str__

tokenized_documents = []
postings = defaultdict(lambda: [])
def prepare(documentDir):
    
    tokenized_documents = prepare_dataset(documentDir)
   # print(tokenized_documents)
    token_docid, doc_ids = get_token_doc_id_pairs(documentDir)
    print(doc_ids)
    sorted_token_docid = sorted(token_docid, key=lambda el: el[0][0])
    merged_tokens_in_doc = merge_token_in_doc(sorted_token_docid)
    dictionary = defaultdict(lambda: (0, 0)) # term : doc_freq, tot freq
   # postings = defaultdict(lambda: []) # term: doc_ids, doc_freq
    for token, doc_id, doc_freq,positions,sentences in merged_tokens_in_doc:
        dictionary[token] = (dictionary[token][0]+1, dictionary[token][1
        ]+doc_freq)

        # usually implemented as linked lists
    for token, doc_id, doc_freq,positions,sentences in merged_tokens_in_doc:
        postings[token].append((doc_id, doc_freq,positions,sentences)) 



    NOT = Keyword("not")
    AND = Keyword("and")
    OR = Keyword("or")
    token = Word(pyparsing_unicode.printables,exclude_chars=punctuation)
    token.setParseAction(BoolRetrievalOperand).setName("token")

    boolOperand = token
    boolOperand.setName("bool_operand")


    boolExpr = infixNotation(
        boolOperand,
        [
            (NOT, 1, opAssoc.RIGHT,BoolRetrievalNot),
            (AND, 2, opAssoc.LEFT,BoolRetrievalAnd),
            (OR, 2, opAssoc.LEFT,BoolRetrievalOr),
        ],
    ).setName("boolean_expression")


    token = Word(alphas)
    token.setParseAction(BoolRetrievalOperand).setName("token")

    wildcard =  Word(alphas) + "*"
    wildcard.setParseAction(BoolRetrievalWildcard).setName("wildcard")

        #   Word(alphas) + ("/" + Word(nums) + Word(alphas))[1,...]
    proximity = Word(alphas) + "/" + Word(nums) + Word(alphas)
    proximity.setParseAction(BoolRetrievalProximity).setName("proximity")

    sentence = "/s" + Word(alphas)[1,...]
    sentence.setParseAction(BoolRetrievalSentence).setName("proximity")

    boolOperand = sentence | proximity | wildcard | token

        # define expression, based on expression operand and
        # list of operations in precedence order
    boolExpr = infixNotation(
        boolOperand,
        [
            (NOT, 1, opAssoc.RIGHT,BoolRetrievalNot),
            (AND, 2, opAssoc.LEFT,BoolRetrievalAnd),
            (OR, 2, opAssoc.LEFT,BoolRetrievalOr),
        ],
    )

    return boolExpr,doc_ids


def test(boolExpr):
    tests = [
        ("t*",True),
        ("one", True),
        ("tree", True),
        ("two /0 tree", True),
        ("two /1 tree", True),
        ("one /4 tree /5 five", True),
        ("/s one two ", True),
        ("/s one two four ", True),
        ("one or tree",True),
        ("(one and (/s one two)) or (will /3 be)",True),
    ]

    for test_string, expected in tests:
        res = boolExpr.parseString(test_string)[0]
        success = "test"#"PASS" if bool(res) == expected else "FAIL"
        print("Query: "+test_string, "\n", res, "=", str(res.process()), "\n", success, "\n")
    
    return 1

def search(doc_ids,searcher,query,maxResults):
    res = searcher.parseString(query)[0]
    unprocessed = res.process()
    if len(unprocessed) > 0:
        unprocessed = map(lambda x: (x[0],x[1]) if len(x) > 2 else x,unprocessed)
    processed = [ doc_ids[doc_id] for doc_id, count in unprocessed  ]
    print(processed)
    return processed[:maxResults]

test(prepare("C:\\Users\\Victor\\OneDrive\\AI\\IR\\IR-NLP-Project\\boolean\\test")[0])