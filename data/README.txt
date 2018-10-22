Each of the data files in this directory is a tsv to be used in modeling.

They are in pairs:
  The one with just the basename is the current training set.
  The one with -test is the current test set.

Note that each week we fold the test set into the training set
and use the next 10% of the raw data as the new test set.

Raw tmdb data is in the tmdb subfolder.

Data that has been partially processed but is not intended to be used
directly in a model is in the intermediate subfolder.

The files in this directory are:

tmdb.tsv - The data from TMDb that is single-valued for each film. This
is the starting point for all our models. It includes the genre_* fields
which are our "Y".

movie-cast-scores.tsv and movie-director-scores.tsv - The processed
genre-affinity scores for each movie. These depend on some intermediate
files but by the time they are in these tsvs each movie has one row,
and there is one column per genre, with the net computed affinity scores.

img_hist.tsv - the very-high-level color information from the poster
images. One row per movie, with columns for the most-frequently seen
pixel values for each channel in both RGB and HSV space. Note that to
model the cyclical nature of hue data, we should model the Hue values
as sin(H/256 * 2pi) and cos(H/256 * 2pi)



To reprocess the data, run scripts from the root cs109 directory
in this order:

python etl/tmdb_to_tsv.py
python etl/tmdb_process_credits.py
python etl/tmdb_credit_tiers.py
python etl/tmdb_director_affinity.py
