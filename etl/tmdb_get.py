''' Initial TMDB step is to use "discover" to find out what movies they have
'''

from json import loads
from time import sleep
from urllib2 import urlopen

api_key = open('apikey.txt').readline().strip()
for year in xrange(2017, 1911, -1):
    # Write each year to its own file
    o = open('data/tmdb/tmdb-' + str(year) + '.txt', 'w')
    page = 1
    # Each response contains a single json "page" object
    # with multiple movies
    # Each file will contain one json object per line
    # We'll parse them later
    # (Except that we need to do minimal parsing here
    # to know when we've reached the last page for the
    # given year)
    while True:
        u = urlopen('https://api.themoviedb.org/3/discover/movie?' +
                    'year=%(year)d&page=%(page)d&api_key=%(api_key)s' %
                {
                    'api_key': api_key,
                    'year': year,
                    'page': page,
                    })
        # If it excepts, we'll let the script die and recover manually
        response = u.read()
        o.write(response + "\n")
        max_page = loads(response)['total_pages']
        print year, ":", page, "/", max_page, "   \r",
        if page == max_page:
            print
            break
        page += 1
        # 250 msec = 4 request/sec = 40 req/10 sec is their rate limit
        sleep(.250) 

