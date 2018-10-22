''' Framework for evaluating models using Spark and Sklearn
'''

import datetime
import numpy as np
import os
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
if 'GridSearchCV' not in globals():
    from sklearn.grid_search import GridSearchCV as call_GridSearchCV
# To use this with Spark, write a wrapper that calls:
#   from spark_sklearn.grid_search import GridSearchCV
# and write a wrapper call_GridSearchSV to provide the Spark Context
# and then import fwork.py

from threading import Thread, Lock

# These globals will be set while loading data
imputed_values = None
train_means = None
train_sd = None
overview_feature_column_names = []

# Record model performance
# Columns are:
# Start_Time, Model_Name, Genre, Base_Rate,
# Train_TruePos, Train_FalsePos, Train_FalseNeg, Train_TrueNeg,
# Test_TruePos, Test_FalsePos, Test_FalseNeg, Test_TrueNeg,
# End_Time, GridSearch_BestParams

columns_to_scale = ['budget', 'runtime', 'revenue',
                    'H_Intensity_0', 'H_Count_0']

genre_affinity_score_column_names = [
    'score_genre_Action',
    'score_genre_Adventure',
    'score_genre_Animation',
    'score_genre_Comedy',
    'score_genre_Crime',
    'score_genre_Documentary',
    'score_genre_Drama',
    'score_genre_Family',
    'score_genre_Fantasy',
    'score_genre_Foreign',
    'score_genre_History',
    'score_genre_Horror',
    'score_genre_Music',
    'score_genre_Mystery',
    'score_genre_Romance',
    'score_genre_Science_Fiction',
    'score_genre_TV_Movie',
    'score_genre_Thriller',
    'score_genre_War',
    'score_genre_Western'
]

def load_dataset(suffix = ''):
    ''' Load all our tsv files and join them in to a
        single master dataframe
    '''
    global imputed_values
    global train_means
    global train_sd

    # Load base dataset
    dataset = pandas.read_table('data/tmdb' + suffix + '.tsv',
                               na_values = 'None',
                               dtype={
                                   'budget': float,
                                   'runtime': float,
                                   'revenue': float
                               })
    print "Basic dataset has", len(dataset), "rows"
    
    # Load and join simple image data
    img_hist = pandas.read_table('data/img_hist' + suffix + '.tsv',
                                 na_values = 'None',
                                 dtype={
                                     'H_Intensity_0': int,
                                     'H_Count_0': int,
                                 })
    dataset = dataset.join(img_hist[['tmdb_id',
                                     'H_Intensity_0', 'H_Count_0']],
                           on='tmdb_id',
                           rsuffix='_r')
    print "After img_hist:", len(dataset), "rows"

    
    # Load and join movie cast genre affinity scores
    affinity = pandas.read_table(
        'data/movie-cast-scores' + suffix + '.tsv',
        na_values = 'None',
    )
    dataset = dataset.join(affinity,
                           on='tmdb_id',
                           rsuffix='_r')

    print "After movie cast affinity:", len(dataset), "rows"
    
    # Load and join dircetor-genre affinity scores
    affinity_director = pandas.read_table(
        'data/movie-director-scores' + suffix + '.tsv',
        na_values = 'None',
    )
    dataset = dataset.join(affinity_director,
                           on='tmdb_id',
                           rsuffix='_director')
    print "After director affinity:", len(dataset), "rows"


    # Load and join columns from IMDB
    imdb = pandas.read_table(
        'data/imdb' + suffix + '.tsv',
        na_values = 'None')
    imdb = imdb.rename(columns= {'IMDB_ID':'imdb_id'})
    imdb['imdb_id'] = 'tt' + imdb['imdb_id'].astype(str)
    imdb = imdb.add_prefix('genre_')
    imdb = imdb.add_suffix('_imdb')
    dataset = pandas.merge(left=dataset, right = imdb, 
                           left_on = "imdb_id",
                           right_on = "genre_imdb_id_imdb",
                           how='left')
    print "After IMDB:", len(dataset), "rows"

    # Load and join cast-genre affinity scores from IMDB data
    affinity_cast_imdb = pandas.read_table(
        'data/imdb-movie-cast-scores' + suffix + '.tsv',
        na_values = 'None',
    )
    affinity_cast_imdb = affinity_cast_imdb.rename(
        columns= {'IMDB_ID':'imdb_id'})
    affinity_cast_imdb['imdb_id'] = (
        'tt' + affinity_cast_imdb['imdb_id'].astype(str)
    )
    affinity_cast_imdb = affinity_cast_imdb.add_prefix('score_genre_')
    affinity_cast_imdb = affinity_cast_imdb.add_suffix('_imdb_cast')
    dataset = pandas.merge(left=dataset, right = affinity_cast_imdb,
                           left_on = "imdb_id",
                           right_on = "score_genre_imdb_id_imdb_cast",
                           how='left')
    print "After IMDB cast affinity:", len(dataset), "rows"

    # Load and join director-genre affinity scores from IMDB data
    affinity_director_imdb = pandas.read_table(
        'data/imdb-movie-director-scores' + suffix + '.tsv',
        na_values = 'None',
    )
    affinity_director_imdb = affinity_director_imdb.rename(
        columns= {'IMDB_ID':'imdb_id'})
    affinity_director_imdb['imdb_id'] = (
        'tt' + affinity_director_imdb['imdb_id'].astype(str)
    )
    affinity_director_imdb = affinity_director_imdb.add_prefix(
        'score_genre_')
    affinity_director_imdb = affinity_director_imdb.add_suffix(
        '_imdb_director')
    dataset = pandas.merge(left=dataset, right = affinity_director_imdb,
                           left_on = "imdb_id",
                           right_on = "score_genre_imdb_id_imdb_director",
                           how='left')
    print "After IMDB director affinity:", len(dataset), "rows"

    # Load and join text features based on overview
    overview_features = pandas.read_table(
        'data/overview_features' + suffix + '.tsv',
        na_values = 'None')
    overview_features = overview_features.add_prefix('overview_')
    global overview_feature_column_names
    overview_feature_column_names = list(overview_features)[1:]
    dataset = pandas.merge(left=dataset, right = overview_features,
                           left_on = "tmdb_id",
                           right_on = "overview_tmdb_id",
                           how='left')
    print "After Overview features:", len(dataset), "rows"


    # Now center and scale the data as appropriate
    
    # print 'Columns:', list(dataset) # column names
    print "Number of rows:", len(dataset)
    if suffix == '':
        print "Means...", datetime.datetime.now()
        if os.path.exists('data/training_means.txt'):
            train_means = pandas.Series(
                eval(open('data/training_means.txt', 'r').read()))
        else:
            train_means = dataset.mean(axis=0)
            with open('data/training_means.txt', 'w') as means_out:
                means_out.write('{')
                for (k, v) in zip(list(train_means.index), train_means):
                    means_out.write("'%s': %e,\n" % (k, v))
                means_out.write('}')
        print "SD...", datetime.datetime.now()
        train_sd = dataset.std(axis=0)
        print "SD cap...", datetime.datetime.now()
        train_sd[train_sd < 0.001] = 1  # Don't let scale blow up
        print "Computing imputed values"
        imputed_values = 0 # dataset.mean(axis=0)
        print imputed_values
    dataset = dataset.fillna(imputed_values, axis=0)
    dataset[columns_to_scale] -= train_means[columns_to_scale]
    dataset[columns_to_scale] /= train_sd[columns_to_scale]
    return dataset



print "Loading movies"
movies = load_dataset()
print "Getting small set"
small_set = movies[movies['tmdb_id'] % 10 == 0]
print "Small set Number of rows:", len(small_set)
smaller_set = movies[movies['tmdb_id'] % 100 == 0]
print "Smaller set Number of rows:", len(smaller_set)
print "Loading test set"
test = load_dataset('-test')
print "Running models"

def conf_matrix(predictions, actual):
    ''' Returns tab-delimited string of TruePos, FalsePos, FalseNeg, TrueNeg
    '''
    true_pos  = ((actual == 1) & (predictions == 1)).sum()
    false_pos = ((actual == 0) & (predictions == 1)).sum()
    false_neg = ((actual == 1) & (predictions == 0)).sum()
    true_neg  = ((actual == 0) & (predictions == 0)).sum()
    return '\t'.join([str(x) for x in
                      [true_pos, false_pos, false_neg, true_neg]])


# For a given model and selection of columns,
# - tune the model using the small subset (~2000 movies)
# - train it on all of this week's training data
# - write out the confusion matrix on both the training and test data sets
def assess_model(name,
                 model,
                 grid_params,
                 genre,
                 columns,
                 use_smaller_set = False,
                 use_base_rate_as_cutoff = False
                ):
    start_time = datetime.datetime.now()

    my_train_set = smaller_set if use_smaller_set else small_set
    
    try:
        # tune on a small set of the data
        gs = call_GridSearchCV(model,
                               param_grid=grid_params,
                               scoring='f1',
        )
        gr = gs.fit(my_train_set[columns],
                    my_train_set['genre_' + genre])
        # then build model on all the training data
        model = gs.best_estimator_
        model.fit(movies[columns],
                  movies['genre_' + genre])
        #with open('data/gs_svc_' + genre + '.txt', 'a') as statfile:
        #    statfile.write(str(gs.cv_results_))
    except Exception as e:
        print name, genre, str(e) 
        with open('data/model_' + genre + '_err.tsv', 'a') as logfile:
            logfile.write(name + str(e))
        return
       
    train_set = movies
    
    if use_base_rate_as_cutoff:
        base_rate = train_set['genre_' + genre].mean()
        train_pred = model.predict_proba(train_set[columns])[:,1] > base_rate
        test_pred = model.predict_proba(test[columns])[:,1] > base_rate
    else:
        train_pred = model.predict(train_set[columns])
        test_pred = model.predict(test[columns])
    
    train_score = model.score(
        train_set[columns],
        train_set['genre_' + genre])
    test_score = model.score(
        test[columns],
        test['genre_' + genre])
    
    print name + ' ' + genre + " Train: %.3f  Test: %.3f  %d or %d test rows" % (
        train_score, test_score, len(test['genre_' + genre]), len(test_pred))

    with open('data/model_' + genre + '_log.tsv', 'a') as logfile:
        logfile.write('\t'.join([str(x) for x in [
            start_time,
            name,
            genre,
            0, # was base_rate -- now we have Dummy model for that
            conf_matrix(train_pred, train_set['genre_' + genre]),
            conf_matrix(test_pred, test['genre_' + genre]),
            datetime.datetime.now(), # endtime
            str(gs.best_params_)
        ]]) + '\n')
                  

def assess_genre(genre):
    assess_model('SVC Fscore',
                 SVC(),
                 {'C': [0.0001, 0.001, 0.01, 1, 100, 1000],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'class_weight': ['balanced',],
                  'kernel': ['rbf',],
                  },
                 genre,
                 ['budget', 'runtime', 'revenue',
                  'H_Intensity_0', 'H_Count_0'] +
                 genre_affinity_score_column_names +
                 overview_feature_column_names,
                 use_smaller_set = True
    )
    
    

# Now iterate over each genre, and for each one, assess each of our models
for genre in [ "Action", "Adventure", "Animation", "Comedy", "Crime",
               "Documentary", "Drama", "Family", "Fantasy", "Foreign",
               "History", "Horror", "Music", "Mystery", "Romance",
               "Science_Fiction", "TV_Movie", "Thriller", "War", "Western", ]:
    Thread(target=assess_genre, name=genre, args=(genre,)).start()

    
