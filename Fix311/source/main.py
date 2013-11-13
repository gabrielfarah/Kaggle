'''
Created on 28/09/2013

@author: Gabrie
'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import BernoulliNB
import re, csv
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import gc
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.neighbors import KNeighborsClassifier

def rmsle(h, y):
    """
   Compute the Root Mean Squared Log Error for hypthesis h and targets y
   
   Args:
       h - numpy array containing predictions with shape (n_samples, n_targets)
       y - numpy array containing targets with shape (n_samples, n_targets)
   """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

def get_city_by_lat_long(longitude):
    if longitude < -100:
        return 1.0 #Oakland
    elif -90 < longitude < -80:
        return 2.0 #Chicago
    elif -80 < longitude < -75:
        return 3.0 #Richmond
    else:
        return 4.0 #New Haven.

def check_below_zero(num):
    if num < 0:
        return 0
    else:
        return num

#This function takes the predicted data and generates the csv response file
def create_csv_response(lenght, ids, pred1, pred2, pred3):
    with open('response.csv', 'wb') as csvout_file:
            csvout = csv.writer(csvout_file)
            csvout.writerows([["id","num_views","num_votes","num_comments"]])
            for resp in range(lenght):
                views = check_below_zero(int(round(pred1[resp])))
                votes = check_below_zero(int(round(pred2[resp])))
                if votes < 1:
                    votes += 1
                comments = check_below_zero(int(round(pred3[resp])))
                csvout.writerows([[ids[resp],views,votes,comments]])
    csvout_file.close()
    print "File created at ./response.csv"

#This function reads the csv file and return the description and the x and y
def read_train_data():
    list_x = []
    list_y_votes = []
    list_y_comments = []
    list_y_views = []
    max_comment_value = 28 #taken from data distribution
    max_view_value = 107 #taken from data distribution
    max_votes_value = 27 #taken from data distribution
    lat = []
    omitted_values = 0
    with open('../data/train.csv', 'r') as content_file:
        content = csv.reader(content_file)
        content.next()
        for row in content:
            try:
                #parse the results in case of errors
                num_votes = int(row[5])
                num_comments = int(row[6])
                num_views = int(row[7])
                val_lat = float(row[2])
                val_lat = get_city_by_lat_long(val_lat)
                #check if this if makes sense...
                if(num_votes <= max_votes_value and num_comments <= max_comment_value and num_views <= max_view_value):
                    description = clear_string(row[4] + " " + row[3])
                    list_x.append(description)
                    list_y_votes.append(num_votes)
                    list_y_comments.append(num_comments)
                    list_y_views.append(num_views)
                    lat.append(val_lat)
                else:
                    omitted_values += 1
            except Exception, e:
                print e
                pass
    content_file.close()
    print "Number of ommited rows given their values: ", omitted_values
    return list_x, list_y_votes, list_y_comments, list_y_views, lat

#This function reads the csv file and return the description and the id from the test data
def read_test_data():
    list_x = []
    list_id = []
    lat = []
    with open('../data/test.csv', 'r') as content_file:
        content = csv.reader(content_file)
        #skip file header
        content.next()
        for row in content:
            try:
                description = clear_string(row[4]+ " "+ row[3])
                id_row = row[0]
                val_lat = float(row[2])
                val_lat = get_city_by_lat_long(val_lat)
                list_x.append(description)
                list_id.append(id_row)
                lat.append(val_lat)
            except Exception, e:
                print e
                pass
    content_file.close()
    return list_x, list_id, lat

#This function removes the numbers from the description
def clear_string(string):
    st = re.sub(r'[0-9]+', '', string)
    return st

def grid_search_model():
    #get data from csv file
    x , y_votes, y_comments, y_views = read_train_data()
    #transform to numpy data type array for better usage
    y_votes = np.array(y_votes)
    y_comments = np.array(y_comments)
    y_views = np.array(y_views)
    #set parameters to try in grid search
    parameters = {
    'vectorizer__max_df': (500, 1000, 3000, 5000),
    'vectorizer__min_df': (1, 2, 3),
    'vectorizer__max_features': (None, 100, 500, 1000),
    #'vectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'vectorizer__use_idf': (True, False),
    'vectorizer__stop_words': ('english', None),
    'clf__alpha': (0.001, 0.01, 0.1, 0.5),
    'clf__fit_prior': (True, False),
    }
    #transformer + classifier pipeline
    classifier = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=None, strip_accents='unicode', ngram_range=(1, 2))),
    ('clf', BernoulliNB(fit_prior=True))])
    grid_search = GridSearchCV(classifier, parameters)
    print"Performing grid search..."
    print"pipeline:", [name for name, _ in classifier.steps]
    print"parameters:"
    print(parameters)
    #Note: USING ONLY VOTES NUMBER TO DO THIS (TEST WITH THE REST ALSO)
    print " Using Votes as train variable.."
    grid_search.fit(x, y_views)
    print "done"
    print 
    print"Best score: %0.3f" % grid_search.best_score_
    print"Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print"\t%s: %r" % (param_name, best_parameters[param_name])

def validate_classifier():
    #get data from csv file
    x , y_votes, y_comments, y_views, lat = read_train_data()
    #transform to nunpy data type array for better usage
    y_votes = np.array(y_votes)
    y_comments = np.array(y_comments)
    y_views = np.array(y_views)
    #split train data to train and test
    x_train_votes, x_test_votes, y_train_votes, y_test_votes = train_test_split(x, y_votes, test_size=0.33)
    x_train_comments, x_test_comments, y_train_comments, y_test_comments = train_test_split(x, y_comments, test_size=0.33)
    x_train_views, x_test_views, y_train_views, y_test_views = train_test_split(x, y_views, test_size=0.33)
    #Change the parameters from the objects with the values from gridsearch
    classifier_votes = Pipeline([('vectorizer', CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2, max_features=None, max_df=1000)),('clf', BernoulliNB(fit_prior=True, alpha=0.001))]) #optimized
    classifier_views = Pipeline([('vectorizer', TfidfVectorizer(stop_words=None, strip_accents='unicode', use_idf=False, ngram_range=(1, 2), min_df=3)),('clf', SGDRegressor(alpha=0.001))]) #optimized
    classifier_comments = Pipeline([('vectorizer', TfidfVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), use_idf=False, min_df=3)),('clf', SGDRegressor(alpha=0.00001))]) #optimized
    print "Fitting model.."
    classifier_votes.fit(x_train_votes, y_train_votes)
    classifier_comments.fit(x_train_comments, y_train_comments)
    classifier_views.fit(x_train_views, y_train_views)
    print "Fitting done"
    print
    predicted_votes = classifier_votes.predict(x_test_votes)
    predicted_comments = classifier_comments.predict(x_test_comments)
    predicted_views = classifier_views.predict(x_test_views)
    #print scores for each value
    print "Votes:"
    print "Model r2 score: %0.3f" % r2_score(y_test_votes, predicted_votes)# best posssible value 1.0
    print "Model mean absolute error: %0.3f" % mean_absolute_error(y_test_votes, predicted_votes) #best value is 0.0
    print
    print "Comments:"
    print "Model r2 score: %0.3f" % r2_score(y_test_comments, predicted_comments)# best posssible value 1.0
    print "Model mean absolute error: %0.3f" % mean_absolute_error(y_test_comments, predicted_comments) #best value is 0.0
    print
    print "Views:"
    print "Model r2 score: %0.3f" % r2_score(y_test_views, predicted_views)# best posssible value 1.0
    print "Model mean absolute error: %0.3f" % mean_absolute_error(y_test_views, predicted_views) #best value is 0.0
    
def perform_emsamble_model():
    #get data from csv file
    x , y_votes, y_comments, y_views, lat = read_train_data()
    #transform to nunpy data type array for better usage
    y_votes = np.array(y_votes)
    y_comments = np.array(y_comments)
    y_views = np.array(y_views)
    #get test data
    x_test, ids, lat = read_test_data()
    #Change the parameters from the objects with the values from gridsearch
    vec_votes = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2)
    vec_comments = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2)
    vec_views = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2)
    #transfor x and x_test in a TFIDF matrix for feeding to the classifier
    x_votes = vec_votes.fit_transform(x)
    x_comments = vec_comments.fit_transform(x)
    x_views = vec_views.fit_transform(x)
    x_test_transformed_votes = vec_votes.transform(x_test)
    x_test_transformed_comments = vec_comments.transform(x_test)
    x_test_transformed_views = vec_views.transform(x_test)
    print "TFIDF Matrixes generated"
    print " LSA transforming"
    lsa_votes = TruncatedSVD(500)
    lsa_comments = TruncatedSVD(500)
    lsa_views = TruncatedSVD(500)
    x_votes = lsa_votes.fit_transform(x_votes)
    print "LSA Votes Done.."
    print
    x_comments = lsa_comments.fit_transform(x_comments)
    print "LSA Comments Done.."
    print
    x_views = lsa_views.fit_transform(x_views)
    print "LSA Views Done.."
    print
    x_test_transformed_votes = lsa_votes.transform(x_test_transformed_votes)
    x_test_transformed_comments = lsa_comments.transform(x_test_transformed_comments)
    x_test_transformed_views = lsa_views.transform(x_test_transformed_views)
    print "SLA Finished.."
    ada_votes = AdaBoostClassifier(base_estimator=RandomForestClassifier())
    ada_comments = AdaBoostClassifier(base_estimator=RandomForestClassifier())
    ada_views = AdaBoostClassifier(base_estimator=RandomForestClassifier())
    ada_votes.fit(x_votes, y_votes)
    ada_comments.fit(x_comments, y_comments)
    ada_views.fit(x_views, y_views)
    print "Fitting done"
    print
    #predict number of votes 
    pred_votes = ada_votes.predict(x_test_transformed_votes)
    pred_comments = ada_comments.predict(x_test_transformed_comments)
    pred_views = ada_views.predict(x_test_transformed_views)
    #generate submission response csv file
    create_csv_response(len(x_test), ids, pred_views, pred_votes, pred_comments)
    
    
def generate_submission_combined_text_lat_long():
    #get data from csv file
    x , y_votes, y_comments, y_views, lat_train = read_train_data()
    #transform to nunpy data type array for better usage
    y_votes = np.array(y_votes)
    y_comments = np.array(y_comments)
    y_views = np.array(y_views)
    lat_train = np.array(lat_train)
    #get test data
    x_test, ids, lat_test = read_test_data()
    lat_test = np.array(lat_test)
    #Change the parameters from the objects with the values from gridsearch
    #vec_votes = TfidfVectorizer(ngram_range=(1, 2), use_idf=False, stop_words=None, strip_accents='unicode', min_df=3, max_features=500, max_df=5000)
    #vec_comments = TfidfVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), use_idf=False, min_df=3)
    #vec_views = TfidfVectorizer(stop_words=None, strip_accents='unicode', use_idf=False, ngram_range=(1, 2), min_df=3)
    vec_votes = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2, max_features=None, max_df=1000)
    vec_comments = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2, max_features=100, max_df=500)
    vec_views = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=1, max_features=None, max_df=500)
    #transfor x and x_test in a TFIDF matrix for feeding to the classifier
    x_votes = vec_votes.fit_transform(x)
    x_comments = vec_comments.fit_transform(x)
    x_views = vec_views.fit_transform(x)
    x_test_transformed_votes = vec_votes.transform(x_test)
    x_test_transformed_comments = vec_comments.transform(x_test)
    x_test_transformed_views = vec_views.transform(x_test)
    print "TFIDF Matrixes generated"
    print
    print "performing data union"
    #X_all_train =  hstack((x.as_matrix(), x_text))
    x_votes =  hstack((lat_train.shape, x_votes))
    x_comments =  hstack((lat_train.shape, x_comments))
    x_views =  hstack((lat_train.shape, x_views))
    x_test_transformed_votes =  hstack((lat_test.shape, x_test_transformed_votes))
    x_test_transformed_comments =  hstack((lat_test.shape, x_test_transformed_comments))
    x_test_transformed_views =  hstack((lat_test.shape, x_test_transformed_views))
    #Create 1 linear classifier for each target variable 
    #num_votes_clf, num_comments_cfl, num_views_cfl = SGDRegressor(alpha=0.00001), SGDRegressor(alpha=0.00001), SGDRegressor(alpha=0.001)
    num_votes_clf, num_comments_cfl, num_views_cfl = BernoulliNB(fit_prior=True, alpha=0.001), BernoulliNB(fit_prior=True, alpha=0.01), BernoulliNB(fit_prior=True, alpha=0.1)
    print "Fitting models.."
    num_votes_clf.fit(x_votes, y_votes)
    num_comments_cfl.fit(x_comments, y_comments)
    num_views_cfl.fit(x_views, y_views)
    print "Fitting done"
    print
    gc.collect()
    #predict number of votes 
    pred_votes = num_votes_clf.predict(x_test_transformed_votes)
    pred_comments = num_comments_cfl.predict(x_test_transformed_comments)
    pred_views = num_views_cfl.predict(x_test_transformed_views)
    #generate submission response csv file
    create_csv_response(len(x_test), ids, pred_views, pred_votes, pred_comments)
    
    
def generate_submission():
    #get data from csv file
    x , y_votes, y_comments, y_views, lat = read_train_data()
    #transform to nunpy data type array for better usage
    y_votes = np.array(y_votes)
    y_comments = np.array(y_comments)
    y_views = np.array(y_views)
    #get test data
    x_test, ids, lat = read_test_data()
    #Change the parameters from the objects with the values from gridsearch
    #vec_votes = TfidfVectorizer(ngram_range=(1, 2), use_idf=False, stop_words=None, strip_accents='unicode', min_df=3, max_features=500, max_df=5000)
    #vec_comments = TfidfVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), use_idf=False, min_df=3)
    #vec_views = TfidfVectorizer(stop_words=None, strip_accents='unicode', use_idf=False, ngram_range=(1, 2), min_df=3)
    vec_votes = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2, max_features=None, max_df=1000)
    vec_comments = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=2, max_features=100, max_df=500)
    vec_views = CountVectorizer(stop_words=None, strip_accents='unicode',analyzer='word',ngram_range=(1, 2), min_df=1, max_features=None, max_df=500)
    #transfor x and x_test in a TFIDF matrix for feeding to the classifier
    x_votes = vec_votes.fit_transform(x)
    x_comments = vec_comments.fit_transform(x)
    x_views = vec_views.fit_transform(x)
    x_test_transformed_votes = vec_votes.transform(x_test)
    x_test_transformed_comments = vec_comments.transform(x_test)
    x_test_transformed_views = vec_views.transform(x_test)
    print "TFIDF Matrixes generated"
    #Create 1 linear classifier for each target variable 
    #num_votes_clf, num_comments_cfl, num_views_cfl = SGDRegressor(alpha=0.00001), SGDRegressor(alpha=0.00001), SGDRegressor(alpha=0.001) KNeighborsClassifier
    num_votes_clf, num_comments_cfl, num_views_cfl = BernoulliNB(fit_prior=True, alpha=0.001), BernoulliNB(fit_prior=True, alpha=0.01), BernoulliNB(fit_prior=True, alpha=0.1)
    print "Fitting models.."
    num_votes_clf.fit(x_votes, y_votes)
    num_comments_cfl.fit(x_comments, y_comments)
    num_views_cfl.fit(x_views, y_views)
    print "Fitting done"
    print
    gc.collect()
    #predict number of votes 
    pred_votes = num_votes_clf.predict(x_test_transformed_votes)
    pred_comments = num_comments_cfl.predict(x_test_transformed_comments)
    pred_views = num_views_cfl.predict(x_test_transformed_views)
    #generate submission response csv file
    create_csv_response(len(x_test), ids, pred_views, pred_votes, pred_comments)

def main(arg):
    if arg =="GS":
        grid_search_model()
    elif arg == "ST":
        validate_classifier()
    elif arg == "response":
        generate_submission()
    elif arg == "ensamble":
        perform_emsamble_model()
    else:
        generate_submission_combined_text_lat_long()
        
if __name__ == '__main__':
    main("ensamble")
