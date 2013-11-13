'''
Created on 30/10/2013

@author: Gabrie
'''
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics,preprocessing,cross_validation
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
    
    
    x = pd.read_csv('../source_data/train.tsv', sep="\t", na_values=['?'], index_col=1)
    x2 = pd.read_csv('../source_data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    y = np.array(x['label'])
    x_text = list(np.array(x['boilerplate']))
    x_test = list(np.array(x2['boilerplate']))
    
    X_all = x_text + x_test
    lentrain = len(x_text)
    
    
    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
              analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
    
    print "fitting and transforming pipeline"
    X_all = tfv.fit_transform(X_all)
    X = X_all[:lentrain]
    x_test = X_all[lentrain:]
    
    print " LSA transforming"
    lsa = TruncatedSVD(400)
    X = lsa.fit_transform(X)
    
    ada = AdaBoostClassifier()
    
    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(ada, X, y, cv=20, scoring='roc_auc'))

    print "training on full data"
    model.fit(X,y)
    pred = model.predict_proba(X_test)[:,1]
    testfile = p.read_csv('../source_data/test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('benchmark.csv')
    print "submission file created.."
