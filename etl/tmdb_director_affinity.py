''' Process the director credits to determine director-genre affinities
    Then use those to compute a per-movie director-genre affinity score
'''

import codecs
from collections import defaultdict

print "Parsing genres"

# First pass: isolate the genre_* columns from our main
# tmdb.tsv table and load them into memory

movie_genres = {}
with codecs.open('data/tmdb.tsv', 'r', 'utf-8') as f:
    tmdb_cols = f.readline().strip().split('\t')
    genre_cols = [c for c in tmdb_cols if c.startswith('genre_')]
    for line in f:
        tmdb_vals = line[:-1].split('\t')
        tmdb_row = dict([z
                         for z in zip(tmdb_cols, tmdb_vals)
                         if z[0].startswith('genre_')])
        movie_genres[int(tmdb_vals[0].strip())] = tmdb_row

# Second pass: For each director, compute the number of times
# they appeared in a movie with each genre (by adding up
# the movie_genres[] rows for each movie they are listed
# as appearing in).

# You can think of the combination of the first two passes
# as bring roughly
#
# SELECT credit.director,
#        SUM(movie.genre_*) as director_genre_*_affinity
#   FROM Credits_By_Name credit
#   JOIN Movie movie
#     ON credit.movie_id = movie_id
#  GROUP BY credit.director

cred_counts = {}
for g in genre_cols:
    cred_counts[g] = defaultdict(int)

print "Parsing credits"

with codecs.open('data/intermediate/director.tsv', 'r', 'utf-8') as f:
    cred_cols = f.readline().strip().split('\t')
    for line in f:
        cred_vals = line[:-1].split('\t')
        cred_name = cred_vals[-1]
        movie_id = cred_vals[0]
        m = movie_genres[int(movie_id)]
        for g in genre_cols:
            cred_counts[g][cred_name] += int(m[g])

print "Parsing credits (round 2)"

# Setup for third pass: Create the 2d array in which
# movie_scores[movie_id][genre_name] = 0
# to hold the genre affinity scores for every
# cartesian combination of movie_id and genre_name

movie_scores = {}
for movie_id in movie_genres.keys():
    movie_scores[movie_id] = {}
    for g in genre_cols:
        movie_scores[movie_id][g] = 0

# Define the minimum number of movies of a particular
# genre in which an director must have worked for us
# to consider their genre affinity to be worth
# tabulating

threshold = 12

# Third pass: For each movie, accumulate the individual
# director-genre affinity scores for every director credited
# as appearing in that movie (provided that the
# director-genre affinity score exceeds the threshold)

with codecs.open('data/intermediate/director.tsv', 'r', 'utf-8') as f:
    cred_cols = f.readline().strip().split('\t')
    for line in f:
        cred_vals = line[:-1].split('\t')
        cred_name = cred_vals[-1]
        movie_id = cred_vals[0]
        for g in genre_cols:
            if cred_counts[g][cred_name] > threshold and cred_counts[g][cred_name] > movie_scores[int(movie_id)][g]:
                movie_scores[int(movie_id)][g] += cred_counts[g][cred_name]

# Write the output.

# Our primary output file is a tsv that can be joined with
# the main tmdb.tsv file. It contains one row per movie,
# and then (in addition to the tmdb_id join key) one
# column per genre, with the movie's cast-genre affiliation
# score as the value.

# NB: since this Python script depends on the tmdb.tsv file,
# that joining *must* be done in the processing environment.
# For example, in R:
#
# tmdb <- read.table('data/tmdb.tsv',
#                    header=T, sep='\t',
#                    encoding='utf-8',
#                    quote=NULL, comment='')
# cast_affinity <- read.table('data/movie-cast-scores.tsv',
#                             header=T, sep='\t')
# tmdb_merged <- merge(tmdb, cast_affinity, by='tmdb_id')

print "Writing affinity scores"                    
with codecs.open('data/movie-director-scores.tsv', 'w', 'utf-8') as o:
    o.write('tmdb_id\t' + '\t'.join(['score_' + g
                                     for g
                                     in genre_cols]) + '\n')
    for movie_id, movie_score in movie_scores.items():
        o.write(str(movie_id))
        for g in genre_cols:
            o.write('\t%d' % movie_score[g])
        o.write('\n')

        
# A secondary output file.
#
# So that we can visually inspect the results of this
# data wrangling step, both to manually verify that the
# output looks reasonable and so that we can be inspired
# by looking at it to pursue other approaches, we create
# a tsv of the top performers in each genre. In particular,
# we expect that by better understanding this aspect of
# the processed data we can figure out how to approach
# the question of secondary affinity.

print "Writing list of top directors in each genre"        
with codecs.open('data/intermediate/director-tiers.tsv', 'w', 'utf-8') as o:
    o.write('Genre\tCount\tName\n')
    for g in genre_cols:
        print g
        sorted_ones = sorted([(v, k)
                              for (k, v)
                              in cred_counts[g].items()
                              if v>=threshold],
                             reverse=True)[:100]
        for (v, k) in sorted_ones:
            o.write('%s\t%d\t%s\n' % (g, v, k))


# Repeat third pass for test data (using the cast affinities
# based only on training data)

# Third pass: For each movie, accumulate the individual
# director-genre affinity scores for every director credited
# as appearing in that movie (provided that the
# director-genre affinity score exceeds the threshold)

print "Computing test data"

movie_scores = {}
with codecs.open('data/intermediate/director-test.tsv', 'r', 'utf-8') as f:
    cred_cols = f.readline().strip().split('\t')
    for line in f:
        cred_vals = line[:-1].split('\t')
        cred_name = cred_vals[-1]
        movie_id = cred_vals[0]
        if int(movie_id) not in movie_scores.keys():
            movie_scores[int(movie_id)] = {}
            for g in genre_cols:
                movie_scores[int(movie_id)][g] = 0
        for g in genre_cols:
            if cred_counts[g][cred_name] > threshold and cred_counts[g][cred_name] > movie_scores[int(movie_id)][g]:
                movie_scores[int(movie_id)][g] = cred_counts[g][cred_name]

# Write the output.

print "Writing affinity scores"                    
with codecs.open('data/movie-director-scores-test.tsv', 'w', 'utf-8') as o:
    o.write('tmdb_id\t' + '\t'.join(['score_' + g
                                     for g
                                     in genre_cols]) + '\n')
    for movie_id, movie_score in movie_scores.items():
        o.write(str(movie_id))
        for g in genre_cols:
            o.write('\t%d' % movie_score[g])
        o.write('\n')
