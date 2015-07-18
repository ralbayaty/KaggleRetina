import numpy as np
import cv2
import sys
from PIL import Image

if __name__ == '__main__':
    try:
        file_name = sys.argv[1]
    except:
        print("Didn't give me a file...")
        file_name = "Lenna.png"
    def nothing(*arg):
      pass

    cv2.namedWindow('mser')
    cv2.createTrackbar('delta', 'mser', 5, 20, nothing)
    cv2.createTrackbar('min', 'mser', 60, 10000, nothing)
    cv2.createTrackbar('max', 'mser', 14400, 100000, nothing)
    cv2.createTrackbar('pass', 'mser', 0, 1, nothing)

    # img = cv2.imread(file_name,1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = Image.open(file_name)
    print(img.size)
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.thumbnail((1024,1024), Image.ANTIALIAS)
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    while True:
        vis = img.copy()
        delta_val = cv2.getTrackbarPos('delta', 'mser')
        min_val = cv2.getTrackbarPos('min', 'mser')
        max_val = cv2.getTrackbarPos('max', 'mser')
        pass_val = cv2.getTrackbarPos('pass', 'mser')
        mser = cv2.MSER_create()
        mser.setDelta(delta_val)
        mser.setMinArea(min_val)
        mser.setMaxArea(max_val)
        mser.setPass2Only(pass_val)
        regions = mser.detectRegions(gray, None)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))

        cv2.imshow('mser', vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()