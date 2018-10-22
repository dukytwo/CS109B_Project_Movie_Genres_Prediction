''' Do explicit feature extraction from the images
    Eventually we hope to do PCA, probably using
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html
    to keep the memory manageable

    For now, load each image.
    If we're in debugging mode, print some summary information

    Save one row to the tsv for each image, containing:
    * the TMDB_id
    * For each of the R, G, B channels ("bands" in PIL-speak):
      * ... for each of the 5 most prevlaent pixel intensities
        * The number of pixels with that intensity,
        * and what that intensity value (0-255) is

'''
import os
from PIL import Image
from threading import Thread

debugging = False

def process_one_image(filename, o):
    i = Image.open(filename)
    if debugging:
        img_bytes = list(i.getdata())
        print "Size:", i.size
        print "Len:", len(img_bytes)
        print "First 24 pixels", ' '.join(str(b) for b in img_bytes[:24])
        print "Histogram", i.histogram()
        isize = i.size[0] * i.size[1]
        
    b = i.resize((48,48)).convert('RGB').convert('HSV').tobytes()
    o.write('\t'.join([(str(ord(bb))) for bb in b]))

def do_block(start_num, end_num):
    with open('data/img_hsv_' + str(start_num) + '.tsv', 'w') as o:
        # Write TSV header row
        o.write('tmdb_id')
        for hsv in 'hsv':
            for y in range(48):
                for x in range(48):
                    o.write('\t%s_%d_%d' % (hsv, y, x))
        o.write('\n')

        for img_num in xrange(start_num, end_num):
            dir_num = int(img_num / 1000)
            filename = 'data/images/%d/%d.jpg' % (dir_num, img_num)
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                print img_num, '   \r   ',
                o.write("%d\t" % (img_num, ))
                process_one_image(filename, o)
                o.write('\n')

step_size = 50000                
breaks = range(0, 450000, step_size)
for i in breaks:
    Thread(target=do_block, args=(i, i+step_size,)).start()
    
    
