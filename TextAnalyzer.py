#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import string

def tokenize(text):
    '''
    Tokenizes terms, removes new lines, leading/trailing punctuations, filter words with less than 2 characters
    for a given text sequence
    '''
    cleaned_tokens = []
    punc = string.punctuation
    text = text.split(' ')
    
    for idx,x in enumerate(text):
        text[idx] = text[idx].replace('\n', '')        
        text[idx] = text[idx].strip(punc)              
        
        if len(text[idx]) >= 2:                        
            cleaned_tokens.append(text[idx].lower())   
    
    return cleaned_tokens

class Text_Analyzer(object):
    ''' Extract token count, top N tokens, and bigrams for a given text sequence'''
    
    def __init__(self, doc):
        self.token_count = {}
        self.text = doc
        
    def analyze(self):
        token_list = tokenize(self.text)    
        
        for word in set(token_list):
            self.token_count[word] = 0
        for i in token_list:
            self.token_count[i] +=1
        return self.token_count
    
    def topN(self, N):
        sort_tokens = sorted(self.token_count.items(), key = lambda item: -item[1])
        return sort_tokens[:N]
    
    def bigram(self, N):
        tokens = tokenize(self.text)
        pair_list = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens)-1)]
        bigram = [(pair, pair_list.count(pair)) for pair in set(pair_list)]
        bigram_sorted = sorted(bigram, key = lambda item: -item[1])
        return bigram_sorted[:N]


# In[ ]:


if __name__ == "__main__":  
    
    # Test Question 1
    text='''What does "immunity" really mean?
            To scientists, immunity means a resistance to a disease gained 
            through the immune system’s exposure to it, either by infection 
            or through vaccination. But immunity doesn’t always mean complete 
            protection from the virus. 

            How does the body build immunity?
            The immune system has two ways to provide lasting protection: 
            T cells that remember the pathogen and trigger a rapid response, 
            and B cells that produce antibodies — proteins the body makes 
            to fight off a specific pathogen.
            So-called “memory T cells” also stick around. Ideally, they live up 
            to their name and recognize a previously encountered pathogen 
            and either help coordinate the immune system or kill infected cells. 
        ''' 
    print("Test Question 1")
    print(tokenize(text))
    
    # Test Question 2
    print("\nTest Question 2")
    analyzer=Text_Analyzer(text)
    analyzer.analyze()
    print(analyzer.token_count)
    print(analyzer.topN(5))
    
    #3 Test Question 3
    print("\nTest Question 3")
    print(analyzer.bigram(3))

