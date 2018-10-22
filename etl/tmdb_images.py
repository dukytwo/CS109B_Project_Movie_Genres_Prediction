''' Take the detalied info derived from TMDB and extract a single .tsv file
    Sampling can come later
'''
import codecs
from json import loads
import os
from time import sleep
from urllib2 import urlopen

# Following list is from https://api.themoviedb.org/3/genre/movie/list?                
genre_ids = [{"id":28,"name":"Action"},{"id":12,"name":"Adventure"},{"id":16,"name":"Animation"},{"id":35,"name":"Comedy"},{"id":80,"name":"Crime"},{"id":99,"name":"Documentary"},{"id":18,"name":"Drama"},{"id":10751,"name":"Family"},{"id":14,"name":"Fantasy"},{"id":36,"name":"History"},{"id":27,"name":"Horror"},{"id":10402,"name":"Music"},{"id":9648,"name":"Mystery"},{"id":10749,"name":"Romance"},{"id":878,"name":"Science Fiction"},{"id":10770,"name":"TV Movie"},{"id":53,"name":"Thriller"},{"id":10752,"name":"War"},{"id":37,"name":"Western"}]

# Which fields will we want to extract?
fields = ['tmdb_id', 'imdb_id', 'title', 'release_date', 'budget', 'original_language', 'popularity', 'vote_average', 'vote_count']
# The genres will be written as a collection of boolean columns
# TODO: production_companies as factors?

# Some films appear in more than one year's data set, so we need to dedup them
ids_dedup = set()

num_seen = 0

consecutive_errors = 0

try:
    for year in xrange(2017, 1914, -1):
        print year
        for movie_str in open('data/tmdb/tmdb-details-' + str(year) + '.txt', 'r'):
            movie = loads(movie_str)

            # Do the dedup
            if movie['id'] in ids_dedup:
                continue
            ids_dedup.add(movie['id'])

            num_seen += 1

            id_chunk = int(movie['id'] / 1000)
            path = 'data/images/' + str(id_chunk)
            if not os.path.exists(path):
                os.mkdir(path)
            if os.path.exists(path + '/' + str(movie['id']) + '.jpg'):
                continue # already have, don't refetch
            with open(path + '/' + str(movie['id']) + '.jpg', 'wb') as o:
                if 'poster_path' not in movie or movie['poster_path'] is None:
                    print "No poster for", movie['id']
                    print year
                    continue
                try:
                    url = 'https://image.tmdb.org/t/p/w160' + movie['poster_path']
                    u = urlopen(url)
                    img_data = u.read()
                    o.write(img_data)
                    o.close()
                    sleep(.005)
                    consecutive_errors = 0
                except Exception as e:
                    print "For movie", movie['id'], str(e)
                    print year
                    consecutive_errors += 1
                    if consecutive_errors > 20:
                        raise Exception("Too many consecutive errors")
                
finally:
    print "Done! Summarized", num_seen, "movies"


