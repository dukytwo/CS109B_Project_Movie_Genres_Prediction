import numpy as np

for cohort in range(10):
    print "Reading", cohort
    data = np.loadtxt('data/img_rgb_cohort_' + str(cohort) + '.tsv', dtype=int, delimiter='\t', skiprows=1)
    print "Reshaping and scaling", cohort
    data = data[:, 1:].reshape(data.shape[0], 48, 48, 3) / 255.0 
    print "Writing", cohort
    np.save('data/img_rgb_color_' + str(cohort) + 'x.npy', data)
    # Then
    print "Verifying", cohort
    data = np.load('data/img_rgb_color_' + str(cohort) + 'x.npy', mmap_mode='r')
    print len(data)
    print "Done with", cohort

    print "Reading Y", cohort
    labels = np.loadtxt('data/genre_cohorts_' + str(cohort) + '.tsv', dtype=int, delimiter='\t', skiprows=1)
    print "Saving Y", cohort
    np.save('data/genre_cohorts_' + str(cohort) + 'x.npy', labels[:,1:])  # get rid of the tmdb_id and save it pickled


