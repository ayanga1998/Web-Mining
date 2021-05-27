#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import numpy as np  
import pandas as pd
import nltk, re, json, string
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords, wordnet
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

def extract(text):
    '''
    Given any table in text format, the function extracts data (ignoring the first row) and 
    returns a list of tuples containing the corresponding values
    
    INPUT:
        text [str]: A table of values in txt format
    
    OUTPUT:
        tuple_list [list]: A list of tuples containing the values for each table row
    '''
    
    reg = re.findall(r'[a-zA-Z0-9_-]+[a-zA-Z0-9._-].+', text)
    
    values = reg[1:]
    values = [" ".join(item.split()).split() for item in values]
    tuple_list = [(item[0], item[1], item[2], item[3]) for item in values]
    
    return tuple_list

def top_collocation(doc, K):
    '''
    This function takes an input document and does the following:
        - Tokenizes the document (regexp) and finds POS tag of each token
        - Creates bigrams of phrases from the tokens using POS tags
        - Gets the frequency of each bigram that is either an ADJ + Noun or Noun + Noun pattern
        - Returns top K collocations
        
    INPUT:
        doc [string]: any document
        K [int]: The number of top collocations by frequency that are desired
        
    OUTPUT:
        result [list]: List of top K collocations in specific patterns for a document
    '''
    
    pattern = r'\w[\w\',-.]*\w' 
    tokens = nltk.regexp_tokenize(doc, pattern)
    tagged_tokens = nltk.pos_tag(tokens)
    bigrams = nltk.bigrams(tagged_tokens)

    phrases = [ (x[0],y[0]) for (x,y) in bigrams if x[1].startswith(('JJ','NN')) and y[1].startswith('NN')]

    dist = nltk.FreqDist(phrases)
    result = dist.most_common(K)
    
    return result

def tokenize(doc):
    '''
    Tokenizes documents using nltk's regular expression tokenizer
    
    INPUT:
        doc [string]: Any document
    
    OUTPUT:
        A list of tokens that are lemmatized and all punctuations, stop words, etc. are removed and 
        converted to lower case format
    '''
    stop_words = stopwords.words('english')
    pattern=r'\w[\w\',-.]*\w' 
    token =nltk.regexp_tokenize(doc.lower(), pattern)

    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = [wordnet_lemmatizer.lemmatize(item, get_wordnet_pos(item))              for item in token if item not in stop_words]                       
    return tokens

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def compute_tfidf(docs):
    '''
    For a given list of documents, this function does the following:
        - Tokenizes each document and stores the tokens in a list
        - Calculate the frequency distribution for the tokens of each separate document
        - Calculates the smoothed IDF and smoothed TF-IDF
    '''

    doc_tokens = [tokenize(doc) for doc in docs]
    token_count = [nltk.FreqDist(tokens) for tokens in doc_tokens]
    
    # Process all documents to a dictionary of dictionaries
    docs_freq = {idx:freq for idx, freq in enumerate(token_count)}
    
    # Create Document Term Matrix
    dtm = pd.DataFrame.from_dict(docs_freq, orient="index" )
    dtm = dtm.fillna(0)
    dtm = dtm.sort_index(axis=0)
    
    
    tf = dtm.values                          # Convert dtm dataframe to numpy arrays
    doc_len = tf.sum(axis=1)
    tf = np.divide(tf, doc_len[:,None])      # divide dtm matrix by the doc length matrix
    np.set_printoptions(precision=2)         # set float precision to print nicely
    
    df = np.where(tf>0,1,0)                  # Get document frequency
    
    smoothed_idf = np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1
    smoothed_tf_idf = normalize(tf*smoothed_idf)
    
    return smoothed_tf_idf 

def find_solutions(qs, doc):
    '''
    Given a list of questions and a document containing answers to the questions, 
    the function performs the following:
        - Segments the article into sentences
        - Concatenate the questions and sentences and compute the TF-IDF
        - Split the TF-IDF into two submatrices for the questions and sentences respectivley
        - Use cosine similarity to find the sentence with the maximum similarity to a question
    
    INPUT:
        qs [list]: a list of questions
        doc [string]: an article containing potential answers to each question
    
    OUTPUT:
        prints each question along with the selected answer using cosine similarity
    '''
    
    sentences = nltk.sent_tokenize(doc)
    
    total = qs + sentences
    tfidf = compute_tfidf(total)
    
    qslen = len(qs)
    tfidf_q = tfidf[:qslen]
    tfidf_sent = tfidf[qslen:]
    
    similarity=1-pairwise_distances(tfidf_q, Y=tfidf_sent, metric = 'cosine')
    sim_sorted = np.argsort(similarity)[:,::-1][:,0]
    
    for idx, sent_idx in enumerate(sim_sorted):
        print('Question ' + str(idx+1) + ': ' + qs[idx])
        print(10*'-------')
        
        print('Answer: ', sentences[sent_idx])
        print('\n')


# In[9]:


if __name__ == "__main__":  
    
    text='''Symbol   Last Price  Change   % Change   Note
                  BTC-USD  56,212.15   -58.16   -0.10%   Bitcoin 
                  ETH-USD  1,787.79    -53.63   -2.91%   Ether
                  BNB-USD  290.51      +5.81    +2.04%   Binance
                  USDT-USD 1.0003      -0.0004  -0.04%   Tether
                  ADA-USD  1.1187      -0.0528  -4.51%   Cardano
      '''
    
    print("\n=========== Test Q1 ============\n")
    print(extract(text))
    
    data = json.load(open("qa.json","r"))
    article = data["context"]
    qas = data["qas"]
    qs = [item["question"] for item in qas]
    
    print("\n=========== Test Q2 ============\n")
    print(top_collocation(article, 10))
    
    
    print("\n=========== Test Q3 ============\n")
    find_solutions(qs, article)
    
    
    print('''
    ANALYSIS:
    
    It is observed that the system can properly identify the sentence in which an answer to a particular question is located. While the "agent" performs well for a majority of the listed problems,some answers may contain more information than necessary (or too little) based on the length and context of the selected statement. This may cause confusion for the user and may provide more issues when implemented at scale.
    
    The function performs well in determining the cosine similarity between questions and sentences, however it selects the wrong answer on a couple of occassions. For example, Question 2 selected an answer that may infer the correct solution that it is spread from person-to-person, but it does not directly address it. This is because the cosine similarity identified key word similarities in both the question and sentence. A similar scenario played out for Question 5 and it highlights the issues that come with using cosine similarity to solve problems that involve cognitive thinking and contextual understanding. Additionally, Question 13 highlights another weakness in the script as it only selects one out of several listed factors needed to be considered. The system separated the article into sentences so the other factors were cut off. 
    
    In order to improve this model, we must expand the ability of the "agent" to process and condense answers so that unneccessary details are not included. We must also provide the capability to include open ended answers if necessasry to address the problem encountered in Question 13. Perhaps we can install a separate method of selecting answers based on euclidean distance that is only activated if a particular question does not generate a similarity score above a certain threshold. Having an integrated system where multiple methods of analyzation can be implemented would allow us to fine tune the model and maximize its performance
    ''')
    
    

