# cs109b Group 3

# Our git repo is at https://github.com/amgreene/cs109b

Data flow and preprocessing are described in MS1 and MS2. Here are the specific
scripts we ran:

## Acquire data from TMDB

* etl/tmdb_get.py -- Use "discover" to enumerate movies
* etl/tmdb_get_details.py -- use "movie" to get the details for each movie
* etl/tmdb_get_credits.py -- use "movie" to get the credits details for each movie
* etl/tmdb_images.py -- Using the downloaded details, fetch the raw images

## Process the affinity scores

* etl/tmdb_credit_tiers.py -- compute the actor affinities
* etl/tmdb_director_affinity.py -- computer the director affinities
* etl/tmbd_process_credits.py -- second pass over the credits, computing the movie scores

## Combine and convert to tsv for further processing

* etl/tmbd_to_tsv.py -- based on the current week's cohort cutoff, create training and test tsvs
* etl/img_hist_split.py -- convert to HSV and extract the 5 most common values in each channel

## Acquire IMDB data

* etl/imdb_get.ipynb -- Get initial data from IMDB
* etl/imdb_to_tsv.ipynb -- convert IMDB data to tsv

## Natural-language processing

* etl/overview_features_tmdb.py -- Create bag-of-words from TMDb overview
* etl/title_fetaures_tmdb.py -- Create bag-of-words from title

## Image processing

* etl/img_hsv.py -- Resample every image to 48x48, convert to HSV, and write it out as a TSV. The work is spread across many threads to take advantage of multi-core systems, and one output file is generated per thread.
* etl/img_hsv-stage2.py -- Concatenate the output files from the previous stage, but re-split them based on the last digit of the tmdb id
* etl/img_hsv_stage3.py -- Column-bind the genre labels to the end of each row of image data to keep them in sync. This is where the bug occurred that caused us to disregard the first two genres.
 
* etl/img_rgb.py -- Do the same thing but converting to RGB instead of HSV
* etl/img_rgb2.py -- Ditto
* etl/img_rgb_np.py -- The equivalent of stage 3, but this time writing directly to a pickled numpy array
* etl/img_rgb_mirror.py -- Again, but mirroring the image to provide augmented data
* etl/img_rgb2_mirror.py -- Ditto

# Traditional models

* spark/fwork.py -- The framework for assessing all our traditional models. Each thread writes a separate text file with information about the performance of its models
* spark/spark_fwork.py -- The wrapper that actually uses Spark
* spark/gridsearch-*.py -- specific versions of fwork to paralellize our gridsearch across mutiple computers
* spark/fwork_stats.py -- Combine all the individual text files into a single TSV suitable for loading into R for visualization

# CNN models

* keras/cnn_loop.py -- The main loop for building and scoring models. As with all our approaches, this writes lots of small text files that can be collected on a single computer and post-processed. There are several variants of this script so we could divide up the work of collecting data on our models.
* keras/scratch.py -- a fragment of Keras/Python code that can replace the VGG-16-based model code in the cnn_loop.py script
* model_logs/extract_performance.py -- The script that collects the small intermediate text files and produces a TSV suitable for loading into R for visualization


# Changes since Milestones

* MS1 and MS2 are unchanged
* We added "Stratified" dummy model to MS3, to confirm that it matches the base rate (this gives us a baseline for MS4)
* We computed data for different batch sizes, which was not done in time for MS4; we did not update MS4 itself but the data are in MS5.

# Locations of zipped data:

All raw images, zipped: 
https://www.dropbox.com/s/b83v7qorfwadfiz/images.zip?dl=0

NPY files for CNN inputs:
https://www.dropbox.com/s/qf9cahppsvurf5e/npy_data.zip?dl=0

All cohorts, genre labels as TSV, zipped:
https://www.dropbox.com/s/2of1agz1b0wz7f6/genre_cohorts_tsv.zip?dl=0

Image RGB data TSVs, zipped:
https://www.dropbox.com/s/iy4m07rf9k19kvh/img_rgb_cohorts.zip?dl=0

TMDB data used for "traditional" models:
https://www.dropbox.com/s/mdzfb93s6ifv4ph/traditional-traintest-data.zip.tmp?dl=0


# Our YouTube video:
https://www.youtube.com/watch?v=hQ29wbZDMPM&feature=youtu.be
