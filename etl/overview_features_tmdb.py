# This script creates text feaures from overview of movies in tmdb

import pandas
import nltk
# nltk.download() # to dowload the corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import pycountry
from json import loads

dataset = pandas.read_table('data/tmdb.tsv',
                               na_values = 'None')

ids_dedup = set()
train_movies = []
for year in xrange(2017, 1914, -1):
    for movie_str in open('data/tmdb/tmdb-details-' + str(year) + '.txt', 'r'):
        movie = loads(movie_str)

        # Do the dedup
        if movie['id'] in ids_dedup:
            continue
        ids_dedup.add(movie['id'])

        movie['tmdb_id'] = movie['id']
        if movie['tmdb_id'] in dataset['tmdb_id']:
            train_movie = {
                'tmdb_id': movie['tmdb_id'],
                'overview': unicode(movie['overview'])
            }
            train_movies.append(train_movie)

print "Tokenizing movie oveview"
# tokenize movie overview
tokenized_docs = [word_tokenize(doc['overview'].lower()) for doc in train_movies]

print "Removing punctuations"
# remove punctuations
regex = re.compile('[%s]' % re.escape(string.punctuation))
tokenized_docs_no_punctuation = []
for overview in tokenized_docs:
    new_overview = []
    for token in overview: 
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_overview.append(new_token)
    tokenized_docs_no_punctuation.append(new_overview)

print "Removing stopwords"
# remove stopwords from overviews in these languages:
stopwords_all_language = []
nltk_lang = ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian',
       'kazakh', 'norwegian', 'portuguese', 'russian', 'russian', 'spanish', 'swedish']
for lang in nltk_lang:
    stopwords_all_language.extend(stopwords.words(lang))

tokenized_docs_no_stopwords = []
for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords_all_language:
            new_term_vector.append(word)
    tokenized_docs_no_stopwords.append(new_term_vector)

print "Populating most common keywords"
# populate the top 50 keywords by frequency
from collections import Counter
tf = Counter()
for doc in tokenized_docs_no_stopwords:
    for word in doc:
        tf[word] +=1
most_common_words = [word for word, word_count in tf.most_common(50)]
print most_common_words

print "Writing to file."
# write them to file
with open('../data/overview_features.tsv', 'w') as o:
    # Write TSV header row
    o.write('tmdb_id' + '\t')
    o.write('\t'.join(most_common_words))
    o.write('\n')
    
    for idx, doc in enumerate(tokenized_docs):
        o.write(str(train_movies[idx]['tmdb_id']) + '\t')
        o.write('\t'.join(['1' if words in doc else '0' for words in most_common_words]))
        o.write('\n')

# create the same features for test set:
test_dataset = pandas.read_table('data/tmdb-test.tsv',
                               na_values = 'None')
ids_dedup = set()
test_movies = []
for year in xrange(2017, 1914, -1):
    for movie_str in open('data/tmdb/tmdb-details-' + str(year) + '.txt', 'r'):
        movie = loads(movie_str)

        # Do the dedup
        if movie['id'] in ids_dedup:
            continue
        ids_dedup.add(movie['id'])

        movie['tmdb_id'] = movie['id']
        if movie['tmdb_id'] in test_dataset['tmdb_id']:
            test_movie = {
                'tmdb_id': movie['tmdb_id'],
                'overview': unicode(movie['overview'])
            }
            test_movies.append(test_movie)
            
# tokenise overview in test set
test_tokenized_docs = [word_tokenize(doc['overview'].lower()) for doc in test_movies]

# write them to file
with open('data/overview_features-test.tsv', 'w') as o:
    # Write TSV header row
    o.write('tmdb_id' + '\t')
    o.write('\t'.join(most_common_words))
    o.write('\n')
    
    for idx, doc in enumerate(test_tokenized_docs):
        o.write(str(test_movies[idx]['tmdb_id']) + '\t')
        o.write('\t'.join(['1' if words in doc else '0' for words in most_common_words]))
        o.write('\n')

print "Feature creation complete"