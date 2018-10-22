''' Take the detalied info derived from TMDB and extract a single .tsv file
'''
import codecs
from json import loads

# The following list of genres is manually retreived
# from https://api.themoviedb.org/3/genre/movie/list?
# and then cleaned up to fit in 80 columns
genre_ids = [{"id":28,"name":"Action"},
             {"id":12,"name":"Adventure"},
             {"id":16,"name":"Animation"},
             {"id":35,"name":"Comedy"},
             {"id":80,"name":"Crime"},
             {"id":99,"name":"Documentary"},
             {"id":18,"name":"Drama"},
             {"id":10751,"name":"Family"},
             {"id":14,"name":"Fantasy"},
             # Foreign is not in that list but
             # can be deduced from the detailed data
             {"id":10679,"name":"Foreign"},
             {"id":36,"name":"History"},
             {"id":27,"name":"Horror"},
             {"id":10402,"name":"Music"},
             {"id":9648,"name":"Mystery"},
             {"id":10749,"name":"Romance"},
             {"id":878,"name":"Science Fiction"},
             {"id":10770,"name":"TV Movie"},
             {"id":53,"name":"Thriller"},
             {"id":10752,"name":"War"},
             {"id":37,"name":"Western"}]

# Which fields will we want to extract?
fields = ['tmdb_id', 'imdb_id', 'title',
          'release_date', 'budget', 'original_language',
          'popularity', 'vote_average', 'vote_count',
          'runtime', 'revenue']

# The genres will be written as a collection of boolean columns

# TODO: production_companies as factors?


# Train vs. Test:

# We will divide our data by taking the last digit of the movie's id
# If it's 0-4 then it's training data from the get-go
# 5 is test data for MS1, and training data after that
# 6 is test data for MS2, and training data after that
# 7 is test data for MS3, and training data after that
# 8 is test data for MS4, and training data after that
# 9 is test data for the final deliverable
test_digit = 6

# Open the output files and write the header row
o_train = codecs.open('data/tmdb.tsv', 'w', 'utf-8')
o_train.write('\t'.join(fields) +
        '\t' +
        '\t'.join(['genre_' + g['name'].replace(' ', '_')
                   for g in genre_ids]) +
        '\n')

o_test = codecs.open('data/tmdb-test.tsv', 'w', 'utf-8')
o_test.write('\t'.join(fields) +
        '\t' +
        '\t'.join(['genre_' + g['name'].replace(' ', '_')
                   for g in genre_ids]) +
        '\n')


# We'll want to know how many movies we've seen
# (just to report progress)
num_seen = 0

# Some films appear in more than one year's data set,
# so we need to dedup them
ids_dedup = set()

# Because we're running this while the downloads are still happening,
# we expect an "incomplete input" ValueError exception to be raised
# when we hit the end of the input file (which has not yet been
# flushed to disk). So we'll use a try/finally so we can keep track
# of how many movies we've seen so far.
try:
    for year in xrange(2017, 1914, -1):
        for movie_str in open('data/tmdb/tmdb-details-' +
                              str(year) + '.txt', 'r'):
            movie = loads(movie_str)

            # Do the dedup
            if movie['id'] in ids_dedup:
                continue
            ids_dedup.add(movie['id'])

            num_seen += 1

            # If this movie's data is supposed to
            # stay concealed until a later milestone,
            # then ignore it for now.
            movie_id_last_digit = movie['id'] % 10
            if movie_id_last_digit > test_digit:
                continue

            # We want the output column to be called
            # tmdb_id rather than id so it's unambiguous
            # Simplest is to copy the existing column
            # to the name we want, and pretend that's what
            # it was called all along
            movie['tmdb_id'] = movie['id']

            # Convert the list of genres to a set which
            # can later be output as a list of booleans
            # in a consistent order
            movie_genres = set()
            for g in movie['genres']:
                movie_genres.add(g['name'])

            # Choose which output file to use based on
            # whether this is training or test data
            o = o_train
            if movie_id_last_digit == test_digit:
                o = o_test

            # Output this movie's row: First, the list
            # of fields that just get passed through,
            # and then the list of genre booleans.
            o.write('\t'.join([unicode(movie[f])
                               for f in fields]
                              +
                              ['1'
                               if g['name'] in movie_genres
                               else '0'
                               for g in genre_ids]) +
                    '\n')

finally:
    print "Done! Summarized", num_seen, "movies"


