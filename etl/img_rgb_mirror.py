''' Do explicit feature extraction from the images
    Eventually we hope to do PCA, probably using
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html
    to keep the memory manageable


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
        
    b = i.resize((48,48)).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT).tobytes()
    o.write('\t'.join([(str(ord(bb))) for bb in b]))

def do_block(start_num, end_num):
    with open('data/img_rgb_mirror_' + str(start_num) + '.tsv', 'w') as o:
        # Write TSV header row
        o.write('tmdb_id')
        for rgb in 'rgb':
            for y in range(48):
                for x in range(48):
                    o.write('\t%s_%d_%d' % (rgb, y, x))
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
    
    
