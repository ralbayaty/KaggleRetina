from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from PIL import Image, ImageDraw


if __name__ == '__main__':
    try:
        file_name = sys.argv[1]
    except:
        print("Didn't give me a file...")
        file_name = "Lenna.png"
    def nothing(*arg):
      pass

    cv2.namedWindow('censure')
    cv2.createTrackbar('min_scale', 'censure', 1, 10, nothing)
    cv2.createTrackbar('max_scale', 'censure', 7, 20, nothing)
    cv2.createTrackbar('mode', 'censure', 2, 2, nothing)
    cv2.createTrackbar('non_max_threshold', 'censure', 6, 1000, nothing)
    cv2.createTrackbar('line_threshold', 'censure', 10, 100, nothing)
    # Defaults: min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10

    # img = cv2.imread(file_name,1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img1 = Image.open(file_name)
    print(img1.size)
    m = 512
    n = 512
    if img1.size[0] > m or img1.size[1] > n:
        img1.thumbnail((m,n), Image.ANTIALIAS)
    print(img1.size)
    gray = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2GRAY)

    mode_dict = {"0": "DoB", "1": "Octagon", "2": "STAR"}
    last_num_kp = 0
    
    while True:
        vis = gray.copy()
        img = img1.copy()

        min_scale = cv2.getTrackbarPos('min_scale', 'censure')
        max_scale = cv2.getTrackbarPos('max_scale', 'censure')
        if min_scale is 0:
        	min_scale = 1
    	if min_scale + max_scale < 3:
    		max_scale = min_scale + 2
        mode = mode_dict[str(cv2.getTrackbarPos('mode', 'censure'))]
        non_max_threshold = float(cv2.getTrackbarPos('non_max_threshold', 'censure'))/1000
        line_threshold = cv2.getTrackbarPos('line_threshold', 'censure')

        censure = CENSURE(min_scale=min_scale, max_scale=max_scale, mode=mode, 
        					non_max_threshold=non_max_threshold, line_threshold=line_threshold)

        censure.detect(vis)
        num_kp = len(censure.keypoints)
        if last_num_kp != num_kp:
        	print("Number of keypoints: " + str(len(censure.keypoints)))
        	last_num_kp = num_kp
        draw = ImageDraw.Draw(img)
        for i in range(min(num_kp,512)):
        	x1 = censure.keypoints[i,1]
        	y1 = censure.keypoints[i,0]
        	x2 = censure.keypoints[i,1]+2**censure.scales[i]
        	y2 = censure.keypoints[i,0]+2**censure.scales[i]
        	coords = (x1, y1, x2, y2)
        	draw.ellipse(coords, fill = None, outline ='blue')

    	scale_hist = cv2.CalcHist(censure.scales)
    	
    	img = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
    	cv2.imshow('censure', img)
        if 0xFF & cv2.waitKey(500) == 27:
            break
    cv2.destroyAllWindows()