o = open('data/model_@_performance.tsv', 'w')
o.write('\t'.join(['Start_Time', 'Model_Name', 'Genre',
                   'Train_TruePos', 'Train_FalsePos', 'Train_FalseNeg', 'Train_TrueNeg',
                   'Test_TruePos', 'Test_FalsePos', 'Test_FalseNeg', 'Test_TrueNeg',
                   'End_Time', 'Tuned_Params']) + '\n')

for genre in [ "Action", "Adventure", "Animation", "Comedy", "Crime",
               "Documentary", "Drama", "Family", "Fantasy", "Foreign",
               "History", "Horror", "Music", "Mystery", "Romance",
               "Science_Fiction", "TV_Movie", "Thriller", "War", "Western", ]:
    # read in accumulated performance, keeping only the last one
    rows = {}
    with open('data/model_' + genre + '_log.tsv', 'r') as i:
        for line in i:
            cols = line[:-1].split('\t')
            # work around bug in fwork if the conf matrix is 1x1
            if len(cols) < 13:
                continue
            # patch older rows that didn't capture the metadata
            if len(cols) == 13:
                cols += ['{}']
            # drop the base rate which we no longer capture
            # because it can be computed downstream
            cols[3:4] = []
            # capture the name, which is our key
            name = cols[1]
            # reconstruct the line and store it
            rows[name] = '\t'.join(cols)
    for line in rows.values():
        o.write(line + '\n')
        
            
