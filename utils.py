import numpy as np
import cv2


def hard(img, threshold=0.5):
    hard_img = np.zeros_like(img)
    hard_img[img >= threshold] = 1.0

    return hard_img


def img_filter(img, f_size=11, mode='mean'):
    function_dict = {'mean': cv2.blur(img, (f_size, f_size)),
                     'gauss': cv2.GaussianBlur(img, (f_size, f_size), 11),
                     'median': cv2.medianBlur(img, f_size)}
    return function_dict[mode]