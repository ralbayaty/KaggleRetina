from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from PIL import Image, ImageDraw

def draw_keypoints(img, kp, scale):
    draw = ImageDraw.Draw(img)
    # Draw a maximum of 300 keypoints
    for i in range(min(len(scale),300)):
        x1 = kp[i,1]
        y1 = kp[i,0]
        x2 = kp[i,1]+2**scale[i]
        y2 = kp[i,0]+2**scale[i]
        coords = (x1, y1, x2, y2)
        draw.ellipse(coords, fill = None, outline ='white')


if __name__ == '__main__':
    try:
        file_name = sys.argv[1]
    except:
        print("Didn't give me a file...")
        file_name = "Lenna.png"
    def nothing(*arg):
      pass

    # Create sliderbars to change the values of CENSURE parameters online 
    # Defaults: min_scale=1, max_scale=7, mode='DoB', non_max_threshold=0.15, line_threshold=10
    cv2.namedWindow('censure')
    cv2.createTrackbar('min_scale', 'censure', 1, 10, nothing)
    cv2.createTrackbar('max_scale', 'censure', 7, 20, nothing)
    cv2.createTrackbar('mode', 'censure', 2, 2, nothing)
    cv2.createTrackbar('non_max_threshold', 'censure', 6, 1000, nothing)
    cv2.createTrackbar('line_threshold', 'censure', 10, 100, nothing)

    # Read image from file, then inspect the image dimensions
    img = cv2.imread(file_name,1)
    height, width, channels = img.shape

    # Pull the different color channels from the image
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Make a PIL image from each channel so we can use PIL.Iamge.thumbnail to resize if needed
    blue1 = Image.fromarray(blue)
    green1 = Image.fromarray(green)
    red1 = Image.fromarray(red)
    gray1 = Image.fromarray(gray)

    # Check if dimensions are above desired, if so then resize keepig aspect ratio
    m, n = 512, 512
    if height > m or width > n:
        blue1.thumbnail((m,n), Image.ANTIALIAS)
        green1.thumbnail((m,n), Image.ANTIALIAS)
        red1.thumbnail((m,n), Image.ANTIALIAS)
        gray1.thumbnail((m,n), Image.ANTIALIAS)

    # CENSURE related
    mode_dict = {"0": "DoB", "1": "Octagon", "2": "STAR"}
    last_num_kp = 0
    
    while True:
        vis = gray.copy()
        img = img1.copy()

        # Read the values of the sliderbars and save them to variables
        min_scale = cv2.getTrackbarPos('min_scale', 'censure')
        max_scale = cv2.getTrackbarPos('max_scale', 'censure')
        if min_scale is 0:
        	min_scale = 1
    	if min_scale + max_scale < 3:
    		max_scale = min_scale + 2
        mode = mode_dict[str(cv2.getTrackbarPos('mode', 'censure'))]
        non_max_threshold = float(cv2.getTrackbarPos('non_max_threshold', 'censure'))/1000
        line_threshold = cv2.getTrackbarPos('line_threshold', 'censure')


        # Create a CENSURE feature detector
        censure = CENSURE(min_scale=min_scale, max_scale=max_scale, mode=mode, 
        					non_max_threshold=non_max_threshold, line_threshold=line_threshold)

        # Obtain the CENSURE features
        censure.detect(blue1)
        kp_blue, scale_blue = censure.keypoints, censure.scales
        censure.detect(green1)
        kp_green, scale_green = censure.keypoints, censure.scales
        censure.detect(red1)
        kp_red, scale_red = censure.keypoints, censure.scales
        censure.detect(gray1)
        kp_gray, scale_gray = censure.keypoints, censure.scales

        # Print the # of features if it has changed between iterations
        num_kp = len(censure.keypoints)
        if last_num_kp != num_kp:
        	print("Number of keypoints: " + str(len(censure.keypoints)))
        	last_num_kp = num_kp

        # Draw the feature points on the images
        draw_keypoints(blue1, kp_blue, scale_blue)
        draw_keypoints(green1, kp_green, scale_green)
        draw_keypoints(red1, kp_red, scale_red)
        draw_keypoints(gray1, kp_gray, scale_gray)
        

        # Obtain the histogram of scale values
        plt.clf()   # clear the figure from any previous plot
    	scale_hist, bin_edges = np.histogram(censure.scales,max_scale-min_scale, (min_scale,max_scale+1))
    	plt.bar(bin_edges[:-1]-0.5, scale_hist, width = 1)
        plt.show(block=False)
        plt.draw()

        # Show the image with keypoints drawn over
    	image = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
    	cv2.imshow('censure', image)
        if 0xFF & cv2.waitKey(500) == 27:
            break
    cv2.destroyAllWindows()