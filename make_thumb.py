#!/usr/bin/env python
# Batch thumbnail generation script using PIL

import sys
import os.path
from PIL import Image

size = (50, 50)
directory = 'thumbs'


N = len(sys.argv)
print("Progress: 0 %"),
# Loop through all provided arguments
for i in range(1, N):
    try:
        # Attempt to open an image file
        filepath = sys.argv[i]
        image = Image.open(filepath)
    except IOError, e:
        # Report error, and then skip to the next argument
        print "Problem opening", filepath, ":", e
        continue

    # Resize the image
    # image = image.resize(thumbnail_size, Image.ANTIALIAS)
    # image = image.thumbnail(size, Image.ANTIALIAS)
    image.thumbnail(size, Image.ANTIALIAS)
    
    # Split our original filename into name and extension
    (name, extension) = os.path.splitext(filepath)

    try:
        # name = name.split("/",1)[1]
        name = os.path.split(name)[1]
    except ValueError:
        pass
    # print(os.path.split(name))
    # Save the thumbnail as "(original_name)_thumb.png"
    image.save(os.path.join('/home/dick/Documents/Kaggle', 'train', name + '_thumb.jpeg'))
    print('\r Progress: ' + str(float(i)*100/N) + '%'),
print("Finished!")