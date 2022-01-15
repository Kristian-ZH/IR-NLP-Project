sample_bbc_news_sentences = [
    "China confirms Interpol chief detained",
    "Turkish officials believe the Washington Post writer was killed in the Saudi consulate in Istanbul.",
    "US wedding limousine crash kills 20",
    "Bulgarian journalist killed in park",
    "Kanye West deletes social media profiles",
    "Brazilians vote in polarised election",
    "Bull kills woman at French festival",
    "Indonesia to wrap up tsunami search",
    "Tina Turner reveals wedding night ordeal",
    "Victory for Trump in Supreme Court battle",
    "Clashes at German far-right rock concert",
    "The Walking Dead actor dies aged 76",
    "Jogger in Netherlands finds lion cub",
    "Monkey takes the wheel of Indian bus"
]
#basic tokenization
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()
sample_bbc_news_sentences_tokenized = [tokenizer.tokenize(sent) 
                            for sent in sample_bbc_news_sentences]
sample_bbc_news_sentences_tokenized[0]

sample_bbc_news_sentences_tokenized_lower = [[_t.lower() 
                                              for _t in _s] 
                for _s in sample_bbc_news_sentences_tokenized]
print(sample_bbc_news_sentences_tokenized_lower[0])


from nltk.tokenize import sent_tokenize
from collections import defaultdict, Counter
from string import punctuation
import os
def preprocess_document(content):
    """
    Returns a list of tokens for a document's content. 
    Tokens should not contain punctuation and should be lower-cased.
    """
    sentences = sent_tokenize(content)
    tokens = []
    for _sent in sentences:
        sent_tokens = tokenizer.tokenize(_sent)
        sent_tokens = [_tok.lower() for _tok in sent_tokens if _tok not in punctuation]
        tokens += sent_tokens
    
    return tokens

def prepare_dataset(documents_dir):
    """
    Returns list of documents in the documents_dir, where each document is a list of its tokens. 
    
    """
    tokenized_documents = []
    for document in os.listdir(documents_dir):
        with open(os.path.join(documents_dir, document), errors='ignore') as outf:
            tokenized_documents.append(preprocess_document(outf.read()))
    print("Found documents: ", len(tokenized_documents))
    return tokenized_documents      

print(prepare_dataset('data/'))