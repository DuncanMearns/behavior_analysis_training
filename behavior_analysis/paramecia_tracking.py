import cv2
import numpy as np
from scipy import ndimage

from .image_processing import rotate_and_centre_image, contour_info


def image_stimulus_map(img, tracking_info):
    """Generates a map of the visual scene in fish-centred coordinates.

    Parameters
    ----------
    img : np.array
        Frame from video
    tracking_info : pd.Series or dict
        Tracking data for given frame

    Returns
    -------
    cropped : np.array
        Head-centred thresholded cropped image
    """
    h, w = img.shape
    blur = 255 - cv2.GaussianBlur(img.astype('uint8'), (3, 3), 0)
    threshed = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
    centred = rotate_and_centre_image(255 - threshed, np.array(tracking_info['midpoint']) / 2., tracking_info['heading'], 0)
    cropped = centred[h / 4: 3 * h / 4, w / 4: 3 * w / 4]
    return cropped


def find_paramecia(img, image_threshold=5, size_threshold=3):
    """Finds putative paramecia within a stimulus map"""
    filt = ndimage.median_filter(img, 5)
    threshed = ndimage.grey_erosion(filt, 3) > image_threshold
    threshed = threshed.astype('uint8') * 255
    im2, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centres, orientations = zip(*[contour_info(cntr) for cntr in contours if cv2.contourArea(cntr) > size_threshold])
    centres = np.array(centres)
    orientations = np.array(orientations)
    return centres, orientations
