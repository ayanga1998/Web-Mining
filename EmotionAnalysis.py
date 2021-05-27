#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd 
import numpy as np

def analyze_tf(X):
    
    # Calculate the document frequency df for word j
    df = np.copy(X)
    df[np.where(df>0)]=1
    df = df.sum(axis=0)
    
    # Calculate the document frequency
    xsum = X.sum(axis = 1)
    tf = (X.T/xsum).T
    
    # Calculate the term frequency-inverse document frequency
    tf_idf = tf/df
    
    # Print the index of the longest document
    print('Indexes of the longest document: ' + str(np.argmax(xsum)))
    
    # Print the index of words with the top 3 largest df values
    print('Indexes of words with the top 3 largest df values: ' + str(np.argsort(df)[::-1][:3]))
    
    # Print the indexes of words with top 3 largest tf_idf values in the longest document:
    idx = np.argmax(xsum)
    print('Indexes of words with top 3 largest tf_idf values in the longest document: ' + str(np.argsort(tf_idf[idx])[::-1][:3]))
    
    return tf_idf

def emotion_analysis():
    
    # Count the number of samples labeled for each emotion
    print('=== The number of samples labeled for each emotion ===')
    emotion = pd.read_csv('emotion.csv')
    print(emotion.emotion.value_counts())
    print('\n')
    
    # Create a new column called length to store the number of words in the text column
    print('=== min, max, and mean values of sadness, happiness, and text length for each emotion ===')
    emotion['length'] = emotion.text.apply(lambda x: len(x.split(' ')))
    em_pivot = pd.pivot_table(data = emotion, values = ['sadness', 'happiness', 'length'], index = 'emotion', 
                               aggfunc={'sadness': [np.mean, np.max, np.min],
                                        'happiness': [np.mean, np.max, np.min],
                                        'length': [np.mean, np.max, np.min]})
    print(em_pivot)
    print('\n')
    
    # Create a cross tabulation to show the average anxiety score of each emotion and each worry value
    print('=== Cross tabulation of anxiety score by emotion and worry ===')
    em_crosstab = pd.crosstab(index = emotion.emotion, columns = [emotion.worry], values = emotion.anxiety, aggfunc = np.mean)
    print(em_crosstab)
    print('\n')
    return

def find_coocur(x):
    
    cooc = np.matmul(x.T,x)
    row, col = np.diag_indices(len(cooc)) 
    cooc[row,col] = 0
    
    print(cooc)
    


# In[10]:


import numpy as np
import pandas as pd


if __name__ == "__main__":  
    
    # Test Question 1
    print("\n")
    print("=== Test Question 1 ===")
    
    dtm = pd.read_csv("dtm.csv")
    x = dtm.values
    analyze_tf(x)
    
    print("\n")
    print("=== Test Question 2 ===")
    emotion_analysis()
    
    print("\n")
    print("=== Test Question 3 ===")
    print(find_coocur(x))

