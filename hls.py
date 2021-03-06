import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#image = mpimg.imread('test_images/test6.jpg')
def hls_select(img, thresh=(0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output


