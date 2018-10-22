# This script creates text feaures from title of movies in tmdb

import pandas
import nltk
# nltk.download() # to dowload the corpus
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import pycountry

dataset = pandas.read_table('data/tmdb.tsv',
                               na_values = 'None',
                               dtype={
                                   'title': str,
                               })

print "Tokenizing titles"
# tokenize titles
tokenized_docs = [word_tokenize(doc.decode('utf-8').lower()) for doc in dataset["title"].astype(str)]

print "Removing punctuations"
# remove punctuations
regex = re.compile('[%s]' % re.escape(string.punctuation))
tokenized_docs_no_punctuation = []
for title in tokenized_docs:
    new_title = []
    for token in title: 
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_title.append(new_token)
    tokenized_docs_no_punctuation.append(new_title)

print "Removing stopwords"
# remove stopwords from titles in these languages:
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
# populate the top 100 keywords by frequency
from collections import Counter
tf = Counter()
for doc in tokenized_docs_no_stopwords:
    for word in doc:
        tf[word] +=1
most_common_words = [word for word, word_count in tf.most_common(50)] # 50 most common keywords
print most_common_words

print "Writing to file."
# write them to file
with open('data/title_features.tsv', 'w') as o:
    # Write TSV header row
    o.write('tmdb_id')
    for word in most_common_words:
        o.write('\t'+ word)
    o.write('\n')
    
    for idx, doc in enumerate(tokenized_docs):
        o.write(str(dataset['tmdb_id'][idx]))
        for words in most_common_words:
            if words in doc:
                o.write("\t" + '1')
            else:
                o.write("\t" + '0')
        o.write('\n')

# create the same features for test set:
test_dataset = pandas.read_table('data/tmdb-test.tsv',
                               na_values = 'None',
                               dtype={
                                   'title': str,
                               })
# tokenise titles in test set
test_tokenized_docs = [word_tokenize(doc.decode('utf-8').lower()) for doc in test_dataset["title"].astype(str)]

# write them to file
with open('data/title_features-test.tsv', 'w') as o:
    # Write TSV header row
    o.write('tmdb_id')
    for word in most_common_words:
        o.write('\t'+ word)
    o.write('\n')
    
    for idx, doc in enumerate(test_tokenized_docs):
        o.write(str(test_dataset['tmdb_id'][idx]))
        for words in most_common_words:
            if words in doc:
                o.write("\t" + '1')
            else:
                o.write("\t" + '0')
        o.write('\n')

print "Feature creation complete"