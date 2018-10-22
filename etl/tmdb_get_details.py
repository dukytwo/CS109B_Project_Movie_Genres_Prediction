''' Get the detailed records for each movie from TMDB
'''

from time import sleep
from json import loads
from urllib2 import urlopen, HTTPError

api_key = open('apikey.txt').readline().strip()

URL_TEMPLATE = ('https://api.themoviedb.org/3/movie/' +
                '%(movie_id)d?api_key=%(api_key)s')

for year in xrange(2017, 1914, -1):
    # One output file per year, containing each JSON
    # response object on its own line.
    o = open('data/tmdb/tmdb-details-' + str(year) + '.txt', 'w')

    # Read in that year's cached responses to the
    # discover query, and iterate over them:
    for page_str in open('data/tmdb/tmdb-' + str(year) + '.txt', 'r'):
        # Convert the json object with one page
        # of 'discover' responses into a dictionary
        page = loads(page_str)
        
        # And iterate over its 'results' element,
        # which contains a list of movies:
        for movie in page['results']:
            # Print some status so we can track progress
            # while this is running
            print ('\r', year, ":",
                   page['page'], "/", page['total_pages'],
                   'Movieid:', movie['id']),
            try:
                # Fetch the detail record for this movie
                u = urlopen(URL_TEMPLATE %
                            {
                                'api_key': api_key,
                                'movie_id': movie['id'],
                            })
                response = u.read()
                o.write(response + "\n")
            except HTTPError as e:
                print "\nEXCEPTION!"
                print e.code
                print e.reason
                # If we hit the occasional 404 error,
                # we want to ignore and keep going.
                # There's one exception: if we have
                # a bug and bust the rate limit, we
                # need to be responsible citizens and
                # bail immediately!
                if e.code == 429:
                    exit()

            # 250 msec = 4 request/sec = 40 req/10 sec
            # that's TMDB's rate limit
            sleep(.250) 
