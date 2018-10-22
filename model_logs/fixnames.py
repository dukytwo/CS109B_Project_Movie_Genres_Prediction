import os
import os.path

for filename in os.listdir('.'):
    newname = filename
    newname = newname.replace('.h5.h5', '.h5')
    newname = newname.replace('.h5history', '-history')
    newname = newname.replace('-256-256-', '-1024-256-')
    newname = newname.replace('-256-64-', '-xxxxx-')

    print filename, newname, os.rename(filename, newname)
    
