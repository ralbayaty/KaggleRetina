from skimage.feature import CENSURE
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from PIL import Image, ImageDraw

def draw_keypoints(img, kp, scale, max_kp=None):
    if max_kp is None:
        max_kp = len(scale)
    draw = ImageDraw.Draw(img)
    # Draw a maximum of 300 keypoints
    for i in range(min(len(scale),max_kp)):
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
        print("You didn't give me a file, so I'm using Lenna.")
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
    blue_ = Image.fromarray(blue)
    green_ = Image.fromarray(green)
    red_ = Image.fromarray(red)
    gray_ = Image.fromarray(gray)

    # Check if dimensions are above desired, if so then resize keepig aspect ratio
    m, n = 512, 512
    if height > m or width > n:
        blue_.thumbnail((m,n), Image.ANTIALIAS)
        green_.thumbnail((m,n), Image.ANTIALIAS)
        red_.thumbnail((m,n), Image.ANTIALIAS)
        gray_.thumbnail((m,n), Image.ANTIALIAS)

    # CENSURE related
    mode_dict = {"0": "DoB", "1": "Octagon", "2": "STAR"}
    last_num_kp = [0,0,0,0]
    num_kp = [0,0,0,0]
    
    while True:
        blue1 = blue_.copy()
        green1 = green_.copy()
        red1 = red_.copy()
        gray1 = gray_.copy()

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
        num_kp[0] = len(kp_blue)
        num_kp[1] = len(kp_green)
        num_kp[2] = len(kp_red)
        num_kp[3] = len(kp_gray)
        if last_num_kp != num_kp:
        	print("Number of keypoints (blue, green, red, gray): " + str(num_kp))
        	last_num_kp = num_kp

        # Draw the feature points on the images
        draw_keypoints(blue1, kp_blue, scale_blue)
        draw_keypoints(green1, kp_green, scale_green)
        draw_keypoints(red1, kp_red, scale_red)
        draw_keypoints(gray1, kp_gray, scale_gray)
        
        plt.clf()
        plt.subplot(2,3,1), plt.imshow(blue1, 'gray'), plt.axis('off'), plt.title('Blue Channel')
        plt.subplot(2,3,2), plt.imshow(green1, 'gray'), plt.axis('off'), plt.title('Green Channel')
        plt.subplot(2,3,3), plt.imshow(red1, 'gray'), plt.axis('off'), plt.title('Red Channel')
        plt.subplot(2,3,4), plt.imshow(gray1, 'gray'), plt.axis('off'), plt.title('Gray Channel')
        plt.show(block=False)
        plt.draw()

        # Show the image with keypoints drawn over
    	# image = cv2.cvtColor(np.asarray(gray),cv2.COLOR_BGR2RGB)
    	cv2.imshow('censure', np.asarray(gray_))
        if 0xFF & cv2.waitKey(500) == 27:
            break
    cv2.destroyAllWindows()