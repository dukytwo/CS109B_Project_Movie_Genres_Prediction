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
        
    # Find 5 strongest pixel intensities for each band
    for rgb in range(3):
        hist = sorted(zip(i.histogram()[rgb*256:(rgb+1)*256],
                          range(256)),
                      reverse =True)[:5]
        if debugging:
            print "Top histogram for "+ 'RGB'[rgb] + ":", hist
        
        for (pixel_count, pixel_intensity) in hist:
            o.write('\t%d\t%d' % (pixel_count, pixel_intensity))

    hsv_hist = i.convert('RGB').convert('HSV').histogram()
    # Find 5 strongest pixel intensities for each band
    for hsv in range(3):
        hist = sorted(zip(hsv_hist[hsv*256:(hsv+1)*256],
                          range(256)),
                      reverse =True)[:5]
        if debugging:
            print "Top histogram for "+ 'HSV'[hsv] + ":", hist
        
        for (pixel_count, pixel_intensity) in hist:
            o.write('\t%d\t%d' % (pixel_count, pixel_intensity))

with open('data/img_hist.tsv', 'w') as o:
    # Write TSV header row
    o.write('tmdb_id')
    for rgb in range(3):
        for i in range(5):
            o.write('\t%(rgb)s_Count_%(i)d\t%(rgb)s_Intensity_%(i)d' % {
                'rgb': 'RGB'[rgb],
                'i': i
                })
    for hsv in range(3):
        for i in range(5):
            o.write('\t%(hsv)s_Count_%(i)d\t%(hsv)s_Intensity_%(i)d' % {
                'hsv': 'HSV'[hsv],
                'i': i
                })
    o.write('\n')

    for img_num in xrange(450000):
        dir_num = int(img_num / 1000)
        filename = 'data/images/%d/%d.jpg' % (dir_num, img_num)
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print img_num, '   \r   ',
            o.write("%d" % (img_num, ))
            process_one_image(filename, o)
            o.write('\n')
    
