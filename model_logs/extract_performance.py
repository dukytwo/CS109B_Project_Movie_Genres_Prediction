''' Convert history as captured in the txt files 
    into a tidy tsv suitable for ggplot in our report
'''

import os
import os.path
import re

exception_messages = []

o = open('model_perf.tsv', 'w')
o.write('\t'.join([
    'Regularizer',
    'FC1',
    'FC2',
    'genre',
    'variant',
    'epoch',
    'train_loss',
    'train_accuracy',
    'train_fscore',
    'test_loss',
    'test_accuracy',
    'test_fscore'
    ]) + '\n')

genre_names= [
    # missing 2 for now
    'Animation',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Family',
    'Fantasy',
    'Foreign',
    'History',
    'Horror',
    'Music',
    'Mystery',
    'Romance',
    'Science_Fiction',
    'TV_Movie',
    'Thriller',
    'War',
    'Western'
]

nan = 'NaN'

for filename in os.listdir('model_logs'):
    if 'history' not in filename:
        continue
    with open(os.path.join('model_logs', filename), 'r') as f:
        try:
            line_1 = f.readline()
            history = eval(line_1.split('Train conf')[0])
            FC1 = 256  # unless overridden
            FC2 = 256  # unless overridden
            if 'BCE-NoReg' in filename:
                (genre_num, FC1, FC2) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)',
                    filename).groups()
                linreg = 'NoReg'
                variant = 'BCE loss No Reg'
            elif 'BCE' in filename:
                (genre_num, FC1, FC2) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)',
                    filename).groups()
                linreg = 'L1=1e-5'
                variant = 'BCE loss'
            elif 'NoReg' in filename:
                (genre_num, FC1, FC2) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)',
                    filename).groups()
                linreg = 'NoReg'
                variant = 'regularizer'
            elif 'batch128' in filename:
                variant = 'batch128'
            elif 'batch64' in filename:
                variant = 'batch64'
            elif '-l1' in filename:
                (genre_num, FC1, FC2, linreg) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)-(l1_[0-9_e]+)',
                    filename).groups()
                linreg = linreg.replace('e_', 'e-')
                linreg = linreg.replace('_', '.')
                linreg = linreg.replace('l1.', 'L1=')
                linreg = linreg.replace('.0', '')
                linreg = linreg.replace('e-0', 'e-')
                linreg = linreg.replace('e+0', 'e+')
                variant = 'regularizer L1'
            elif '-l2' in filename:
                (genre_num, FC1, FC2, linreg) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)-(l2_[0-9_e]+)',
                    filename).groups()
                linreg = linreg.replace('e_', 'e-')
                linreg = linreg.replace('_', '.')
                linreg = linreg.replace('l2.', 'L2=')
                linreg = linreg.replace('.0', '')
                linreg = linreg.replace('e-0', 'e-')
                linreg = linreg.replace('e+0', 'e+')
                variant = 'regularizer L2'
            elif 'scratch_' in filename:
                (genre_num, FC1, FC2) = re.search(
                    r'scratch_model-(\d+)-(\d+)-(\d+)',
                    filename).groups()
                linreg = 'L1=1e-5'
                variant = 'scratch'
            elif '_LRS_' in filename:
                (genre_num, FC1, FC2) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)_LRS_',
                    filename).groups()
                linreg = 'L1=1e-5'
                variant = 'LR decay'
            elif '_RLROP_100' in filename:
                if 'drama' in filename:
                    genre_num = 4
                elif 'comedy' in filename:
                    genre_num = 1
                elif 'anima' in filename:
                    genre_num = 0
                else:
                    raise Exception("Unknown genre")
                FC1 = '256'
                FC2 = '256'
                linreg = 'L1=1e-5'
                variant = '100 epochs Reduce LR on Plateau'
            elif '_RLROP_' in filename:
                (genre_num, FC1, FC2) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)_RLROP_',
                    filename).groups()
                linreg = 'L1=1e-5'
                variant = 'Reduce LR on Plateau'
            elif '-VGG16-pretrain-' in filename:
                (genre_num, frozen) = re.search(
                    r'model-(\d+)-VGG16-pretrain-(\d+)-history',
                    filename).groups()
                linreg = 'L1=1e-5'
                variant = 'VGG w/' + frozen + ' frozen layers'
            else:
                (genre_num, FC1, FC2) = re.search(
                    r'model-(\d+)-(\d+)-(\d+)',
                    filename).groups()
                if FC1 == '256' and FC2 == '1024':
                    continue # don't have a full set of these
                if FC1 == '1024' and FC2 == '4':
                    continue # don't have a full set of these
                linreg = 'L1=1e-5'
                variant = 'base'
            genre = genre_names[int(genre_num)]
            print FC1, FC2, genre
            for epoch in range(len(history['loss'])):
                o.write('\t'.join([str(v) for v in [
                    linreg,
                    FC1,
                    FC2,
                    genre,
                    variant,
                    epoch + 1,
                    history['loss'][epoch],
                    history['binary_accuracy'][epoch],
                    history['fscore'][epoch],
                    history['val_loss'][epoch],
                    history['val_binary_accuracy'][epoch],
                    history['val_fscore'][epoch],
                ]]) + '\n')
        except Exception as e:
            print filename, e
            exception_messages += [filename + ' ' + str(e),]

print "DONE!"            
print exception_messages
