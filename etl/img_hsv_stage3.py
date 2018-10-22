genres= {}
infile = open('data/tmdb.tsv', 'r')
genres_head = '\t'.join(infile.readline().split('\t')[-18:])
for line in infile:
    s = line.split('\t')
    tmdb_id = int(s[0])
    genres[tmdb_id] = '\t'.join(s[-18:])

zeros = '\t'.join(['0' for i in range(18)]) + '\n'

for cohort in ['00'] + [str(c) for c in range(6)]:  # can't go past current training set yet....
    o = open('data/genre_cohorts_' + cohort + '.tsv', 'w')
    o.write(genres_head)
    infile = open('data/img_rgb_cohort_' + cohort + '.tsv', 'r')  # just to get ids in the right order
    infile.readline()
    for line in infile:
        tmdb_id = int(line.split('\t')[0])
        if not genres.has_key(tmdb_id):
            print "Missing tmdb_id", tmdb_id
        o.write('%d\t%s\n' % (tmdb_id, genres.get(tmdb_id, zeros)))
