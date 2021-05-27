#!/usr/bin/env python
# coding: utf-8

# In[17]:


from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support, classification_report
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pd 
import numpy as np

def predict(X_train, y_train, x_test, y_test, model_type, C):
    if model_type == 'nb':
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        predict_p = clf.predict_proba(x_test)
        y_pred_prob = predict_p[:,1]
        y_pred = clf.predict(x_test)
    elif model_type == 'svm':
        clf = svm.LinearSVC(C=C)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.decision_function(x_test)
        y_pred = clf.predict(x_test)
    
    return y_pred, y_pred_prob

def AUC_PRC(y_test, y_pred, y_pred_prob, model_type, print_result):
    if model_type == 'nb':
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 1)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label=1)
        
        title = 'Naive Bayes Model'
        
    elif model_type == 'svm':
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label=1)
    
        title = 'Support Vector Machine'
    
    # Plot AUC 
    if print_result == True:
        print(classification_report(y_test, y_pred, target_names = ['0','1']))
        
        plt.figure()
        plt.plot(fpr, tpr, color = 'darkorange', lw = 2)
        plt.plot([0,1],[0,1], color = 'navy', lw=2, linestyle = '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC of ' + title)
        plt.show()
    
        # Plot PRC
        plt.figure()
        plt.plot(recall, precision, color = 'darkorange', lw = 2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision_Recall_Curve of ' +  title)
        plt.show()
    else:
        pass
    
    AUC = auc(fpr, tpr)
    PRC = auc(recall, precision)
    
    return AUC, PRC

def create_model(train_docs, train_y, test_docs, test_y, model_type, smooth_idf=False, stop_words=None, min_df = 1, print_result = True, C = 1, norm = 'l2'):

    vectorizer = TfidfVectorizer(stop_words = stop_words, min_df=min_df, norm = norm, smooth_idf=smooth_idf)
    
    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)
    
    y_pred, y_pred_prob = predict(X_train, train_y, X_test, test_y, model_type, C)
    auc_score, prc_score = AUC_PRC(test_y, y_pred, y_pred_prob, model_type, print_result)
    print("AUC: {}, PRC: {}".format(auc_score, prc_score))
    
    return auc_score, prc_score

def search_para(docs, y):
    
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', svm.LinearSVC())])
    parameters = {'tfidf__min_df':[1, 2,5],
              'tfidf__stop_words':[None,"english"],
                  'tfidf__smooth_idf':[True, False],
                  'tfidf__norm':['l1', 'l2'],
                 'clf__C':[0.01, 1, 10, 100]
                 }

    # the metric used to select the best parameters
    metric =  "f1_macro"

    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)
    gs_clf = gs_clf.fit(docs, y)

    for param_name in gs_clf.best_params_:
        print("{0}:\t{1}".format(param_name, gs_clf.best_params_[param_name]))

    print("best f1 score: {:.3f}".format(gs_clf.best_score_))

def sample_size_impact(train_docs, train_y, test_docs, test_y):   
    
    df_svm = pd.DataFrame(columns= ['sample_size', 'AUC', 'PRC'])
    df_nb = pd.DataFrame(columns= ['sample_size', 'AUC', 'PRC'])
    x0 = 500

    for i in range(int(np.sqrt(len(train)))):
        x = x0 + i*500
        
        if x < len(train):
            
            size = round(x/len(train),ndigits = 5)
            X_train = train_docs[0:x]
            y_train = train_y[0:x]
      
            # implement for SVM
            auc_score, prc_score = create_model(X_train, y_train, test_docs, test_y, model_type='svm', stop_words = 'english', min_df = 1, print_result=False)
            dict_svm = {'sample_size': x, 'AUC':auc_score, 'PRC': prc_score}
            df_svm = df_svm.append(dict_svm, ignore_index = True)
        
            # implement for Naive Bayes
            auc_score, prc_score = create_model(X_train, y_train, test_docs, test_y, model_type='nb', stop_words = 'english', min_df = 1, print_result=False)
            dict_nb = {'sample_size': x, 'AUC':auc_score, 'PRC': prc_score}
            df_nb = df_nb.append(dict_nb, ignore_index = True)
            
    # Plot the AUC scores
    plt.plot(df_nb.sample_size, df_nb.AUC, c='darkorange', label = 'Naive Bayes')
    plt.plot(df_svm.sample_size, df_svm.AUC, c='navy', label = 'SVM')
    plt.title('AUC Scores Based on Sample Size and Method')
    plt.xlabel('Sample Size')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[18]:


if __name__ == "__main__":  
     
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Test Q1
    print("Q1")
    auc_score, prc_socre = create_model(train["text"], train["label"], test["text"], test["label"],           model_type='svm', stop_words = 'english', min_df = 1, print_result=True)
    
    auc_score, prc_socre = create_model(train["text"], train["label"], test["text"], test["label"],           model_type='nb', stop_words = 'english', min_df = 1, print_result=True)
    
    # Test Q2
    '''
    Note: Grid search best parameters
    C: 10
    min_df: 2
    norm: l1
    smooth_idf: True
    stop_words: None
    '''
    print("\nQ2")
    search_para(train["text"], train["label"])
    auc_score, prc_score = create_model(train["text"], train["label"], test["text"], test["label"],           model_type='svm', stop_words = None, min_df = 1, print_result=True)
    
    # Test Q3
    print("\nQ3")
    sample_size_impact(train["text"], train["label"], test["text"], test["label"])
    
    # Test Q4 
    print("\nQ4")
    auc_score, prc_score = create_model(train["text"], train["label"], test["text"], test["label"],           model_type='svm', stop_words = None, min_df = 2, print_result=True, smooth_idf=True, C = 10,
                                   norm = 'l1')


# In[ ]:




