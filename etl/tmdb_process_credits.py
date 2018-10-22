''' Take the credits data and convert it to tsvs for cast and director
'''
import codecs
from collections import defaultdict
from json import loads

# Cast
cast_train = defaultdict(list)
cast_test = defaultdict(list)
cast_names = {}

# Some films appear in more than one year's data set, so we need to dedup them
ids_dedup = set()

# We will divide our data by taking the last digit of the movie's id
# If it's 0-4 then it's training data from the get-go
# 5 is test data for MS1, and training data after that
# 6 is test data for MS2, and training data after that
# 7 is test data for MS3, and training data after that
# 8 is test data for MS4, and training data after that
# 9 is test data for the final deliverable
test_digit = 6

o_director_train = codecs.open('data/intermediate/director.tsv', 'w', 'utf-8')
o_director_train.write('Movie_ID\tDirector_ID\tDirector_Name\n')
o_director_test = codecs.open('data/intermediate/director-test.tsv', 'w', 'utf-8')
o_director_test.write('Movie_ID\tDirector_ID\tDirector_Name\n')

for year in xrange(2017, 1914, -1):
    print year,
    for movie_str in open('data/tmdb/tmdb-credits-' + str(year) + '.txt', 'r'):
        movie = loads(movie_str)

        movie_id = movie['id']
        
        # Do the dedup
        if movie_id in ids_dedup:
            continue
        ids_dedup.add(movie_id)

        movie_id_last_digit = movie_id % 10
        if movie_id_last_digit > test_digit:
            continue
        cast = cast_train
        o_director = o_director_train
        if movie_id_last_digit == test_digit:
            cast = cast_test
            o_director = o_director_test

        cast_dedup = set()
        for cast_member in movie['cast']:
            # Ignore when one actor has multiple roles in one movie
            # e.g. Sellars in Strangelove
            if cast_member['id'] in cast_dedup:
                continue
            cast_dedup.add(cast_member['id'])
            cast[cast_member['id']].append(movie['id'])
            cast_names[cast_member['id']] = cast_member['name'].strip().replace('\t', ' ')

        for crew_member in movie['crew']:
            if crew_member['job'] == 'Director':
                o_director.write(unicode(movie['id']) + '\t' +
                                 unicode(crew_member['id']) + '\t' +
                                 crew_member['name'].strip() + '\n')
                
cast_tuples = sorted([(len(movie_ids),
                       cast_names[cast_id],
                       cast_id,
                       movie_ids)
                      for (cast_id, movie_ids)
                      in cast_train.items()])

o = codecs.open('data/intermediate/credits-by-name.txt', 'w', 'utf-8')
o.write('Num_roles\tName\tCast_id\tMovie_id_list\n')
for (len_roles, cast_name, cast_id, movie_ids) in cast_tuples:
    o.write('%(len_roles)d\t%(cast_name)s\t%(cast_id)d\t%(movie_ids)s\n' % {
        'len_roles': len_roles,
        'cast_name': cast_name,
        'cast_id': cast_id,
        'movie_ids': ','.join([str(i) for i in movie_ids])
        })


cast_tuples = sorted([(len(movie_ids),
                       cast_names[cast_id],
                       cast_id,
                       movie_ids)
                      for (cast_id, movie_ids)
                      in cast_test.items()])

o = codecs.open('data/intermediate/credits-by-name-test.txt', 'w', 'utf-8')
o.write('Num_roles\tName\tCast_id\tMovie_id_list\n')
for (len_roles, cast_name, cast_id, movie_ids) in cast_tuples:
    o.write('%(len_roles)d\t%(cast_name)s\t%(cast_id)d\t%(movie_ids)s\n' % {
        'len_roles': len_roles,
        'cast_name': cast_name,
        'cast_id': cast_id,
        'movie_ids': ','.join([str(i) for i in movie_ids])
        })
    
