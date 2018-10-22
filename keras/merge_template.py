# Template for merging two models (branches)
#
# Note: Will get the following warning at runtime:
# The `Merge` layer is deprecated and will be removed after 08/2017
#  
#
#  Create the metadata branch for non-image features
#  Additional layers can be added as needed
#
metadata_branch = Sequential()
metadata_branch.add(Dense(metadata.shape[1], input_shape = (metadata.shape[1],), init = 'normal', activation = 'relu'))
#
# Insert CNN branch model here
# Can use existing scratch or fine tuned CNN model 
# and output the desired number of features to merge 
#
# Merge the metadata and CNN branches via Merge, use 'concat' mode
#
model = Sequential()
model.add(Merge([metadata_branch1, cnn_branch2], mode = 'concat'))
#
#  Add additional layers post merge 
#
#  When fitting, need to include the data sources from both branches
#  For example, using: 
#	train datasources:
#		 "metadata" and "imagedata"
#       validation datasources:
#		 "validation_metadata" and "validation_imagedata"

history = model.fit([metadata, imagedata],     # training data
                    labels,   # training labels
#                   Include other parameters as needed
                    validation_data=
                    ([validation_metadata, validation_imagedata],
                     validation_labels),
                    )
