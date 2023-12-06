import json
from rank_bm25 import *
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords


ps=PorterStemmer()



with open("Project/Lamp4-copy.json","r") as file:
    data=json.load(file)
    

    for user in data:
        tokenized_corpus=[]
        corpus=[]
        
        ip=user['input']
        profile=user['profile']

        #create corpus for each user
        for feature in profile:
            text=feature['text']
            corpus.append(text)

            # tokenize the corpus for each user
            tokenized_text = text.split()

            tokenized_text=[i for i in tokenized_text if str(i) not in string.punctuation]              #remove special characters
            tokenized_text=[word for word in tokenized_text if word not in stopwords.words('english')]  #remove stopwords
            tokenized_text=[ps.stem(word) for word in tokenized_text]                                   #stemm tokenize corpus
            tokenized_corpus.append(tokenized_text) 


        #print("corpus:",corpus)
        #print("tokenized_corpus:",tokenized_corpus)

        #indexing the data
        bm25=BM25Okapi(tokenized_corpus)
        tokenized_q=ip.split(" ")
        doc_scores = bm25.get_scores(tokenized_q)
        print("doc_scores:",doc_scores)
        top_n=bm25.get_top_n(tokenized_q, corpus, n=3)
        print("query:",ip)
        print("top_n docs are:",top_n)

#print("corpus:",corpus)







