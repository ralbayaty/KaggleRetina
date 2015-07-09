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

    cv2.namedWindow('freak')
    cv2.createTrackbar('hess', 'freak', 5, 20, nothing)

    # img = cv2.imread(file_name,1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = Image.open(file_name)
    print(img.size)
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.thumbnail((1024,1024), Image.ANTIALIAS)
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    while True:
        hess_val = cv2.getTrackbarPos('hess', 'freak')
        # freak = cv2.DescriptorExtractor_create('FREAK')
        freak = cv2.xfeatures2d.FREAK_create()
        # help(freak)
        # freak.setHessian(hess_val)
        kp,des = freak.detectAndCompute(gray,None)
        img2 = cv2.drawKeypoints(gray,kp,gray)
        cv2.imshow('freak', img2)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()