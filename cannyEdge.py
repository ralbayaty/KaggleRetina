
#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [path/file_name.ext]

  Trackbars control edge thresholds.

'''

import cv2
import numpy as np
import sys
from PIL import Image


if __name__ == '__main__':
    # print __doc__

    try:
        file_name = sys.argv[1]
    except:
        print("Didn't give me a file...")
        file_name = "Lenna.png"

    def nothing(*arg):
      pass

    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 1024, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 150, 5000, nothing)

    # img = cv2.imread(file_name,1)
    # img = cv2.resize(img, (256,256))
    img = Image.open(file_name)
    img.thumbnail((1024,1024), Image.ANTIALIAS)
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        vis = img.copy()
        vis /= 2
        vis[edge != 0] = (0, 255, 0)

        cv2.imshow('edge', vis)
        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:
            break
    cv2.destroyAllWindows()




