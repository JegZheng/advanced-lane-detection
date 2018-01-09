import numpy as np
import pickle
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import skimage
import skimage.transform
from sobel import combined_thresh
from perspective_transform import perspective_transform
from sliding_windows import draw_lines, line_fit, calc_curve, calc_vehicle_offset, final_viz
import glob



def process_image(img_in):
    global mtx, dist, left_line, right_line, detected
    global left_curve, right_curve, left_lane_inds, right_lane_inds

    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    if img_in.shape[0] == 720 and img_in.shape[1] == 1280:
        region = [np.float32(
            [[200, 720],
            [1100, 720],
            [595, 450],
            [698, 450]]) , np.float32(
            [[300, 720],
            [980, 720],
            [300, 0],
            [980, 0]])]
    else:
        region = [np.float32(
            [[400, 375],
            [900, 375],
            [595, 205],
            [685, 205]]) , np.float32(
            [[300, 375],
            [980, 375],
            [300, 0],
            [980, 0]])]
    # Undistort, threshold, perspective transform
    undist = cv2.undistort(img_in, mtx, dist, None, mtx)
    img = combined_thresh(undist)
    binary_warped, binary_unwarped, m, m_inv = perspective_transform(img,region)

    # Perform polynomial fit

    # Slow line fit
    ret = line_fit(binary_warped)
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']


    # Calculate curvature
    left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

    vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

    result = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)
    return result



img_file = glob.glob('test_images/*')
#print img_file

for img_ in img_file:
    print img_
    if img_.split('.')[-1] == 'png':
        img = mpimg.imread(img_)
    else:
        img = mpimg.imread(img_)
        shape_ref = img.shape


    result = process_image(img)
#print result
    plt.imshow(result)
    plt.savefig('output/'+ img_.split('/')[1])
#plt.show()


