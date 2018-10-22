''' Summarize genre information from TMDB discover data
'''

from collections import defaultdict
from itertools import combinations
from json import loads

# We'll accumulate both singleton and doubleton data
# That is, each movie may have multiple genres associated with it
# We will record how many times each genre appears (genres)
# And we will record how many times each genre appears paired with another
# For now we won't worry about triples, etc.
genres = defaultdict(int)
genre_pairs = defaultdict(int)

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

gnames = {}
for d in genre_ids:
    gnames[d['id']] = d['name']

# Note that not all genre_id numbers are found in the list we retrieved
def genre_name(genre_id):
    return gnames.get(genre_id, 'Genre #'+str(genre_id))

# Because we had to download the discovery data one year at a time,
# and because TMDB can associate a movie with more than one year,
# we need to maintain a list of ids that we've already seen so we
# can avoid double-counting
ids_dedup = set()

# Now iterate over the data we've collected
for year in xrange(2017, 1914, -1):
    for page in open('data/tmdb/tmdb-' + str(year) + '.txt', 'r'):
        for movie in loads(page)['results']:
            # Check whether we've already seen this movie
            if movie['id'] in ids_dedup:
                continue
            ids_dedup.add(movie['id'])
            my_genres = sorted([genre_name(g_id)
                                for g_id in movie['genre_ids']])
            # Tabluate its genres
            for genre in my_genres:
                genres[genre] += 1
            # Tabulate its genre co-incidences
            for genre_pair in combinations(my_genres, 2):
                genre_pairs[genre_pair] += 1

# Print column headers
print "Genre_1\tGenre_2\tNum_Movies"

# Print out the singleton counts
for genre, count in sorted(genres.items()):
    print genre, '\t\t', count

# separate singletons from pairs    
print

# Print out the doubleton counts
for genres, count in sorted(genre_pairs.items()):
    print '\t'.join(genres) + '\t' + str(count)

    
    
