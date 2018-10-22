o = [open('data/img_hsv_cohort_' + str(i) + '.tsv', 'w') for i in range(10)]
o00 = open('data/img_hsv_cohort_00.tsv', 'w')

step_size = 50000                
breaks = range(0, 450000, step_size)
for i in breaks:
    print '\r' + str(i),
    infile = open('data/img_hsv_' + str(i) + '.tsv' , 'r')
    first_line = infile.readline()
    if i == 0:
        for oo in o:
            oo.write(first_line)
        o00.write(first_line)
        num_cols = len(first_line.split('\t'))
    for line in infile:
        s = line.split('\t')
        if len(s) != num_cols:
            continue
        tmdb_id = int(s[0])
        o[tmdb_id % 10].write(line)
        if tmdb_id % 100 == 0:
            o00.write(line)
