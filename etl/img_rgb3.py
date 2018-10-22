genre_sf= {}
infile = open('data/tmdb.tsv', 'r')
infile.readline()
for line in infile:
    s = line.split('\t')
    tmdb_id = int(s[0])
    if tmdb_id % 100 != 0:
        continue
    genre_sf[tmdb_id] = s[-5]

# feh    
o = open('data/genre_cohort.tsv', 'w')

infile = open('data/img_hsv_cohort_00.tsv', 'r')
infile.readline()
for line in infile:
    tmdb_id = int(line.split('\t')[0])
    o.write('%d\t%s\n' % (tmdb_id, genre_sf[tmdb_id]))
