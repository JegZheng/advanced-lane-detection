import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sobel import combined_thresh


def perspective_transform(img , region):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    '''
    src = np.float32(
        [[200, 720],
        [1100, 720],
        [595, 450],
        [685, 450]])
    dst = np.float32(
        [[300, 720],
        [980, 720],
        [300, 0],
        [980, 0]])
    '''
    src = region[0]
    dst = region[1]
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

    return warped, unwarped, m, m_inv


