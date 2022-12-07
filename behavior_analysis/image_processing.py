import cv2
import numpy as np


def rotate_and_centre_image(image, centre, angle, fill_outside=255):
    """Rotates and centres an image

    Parameters
    ----------
    image : array-like
        An image represented as an array

    centre : tuple, list or array-like
        The point in the image that is to become the centre (x, y) coordinates

    angle : float
        The angle through which to rotate the image (radians)

    fill_outside : int, optional (0-255)
        The grayscale value to fill points outside the original image

    Returns
    -------
    stabilised : array
        The image centred on the given point and rotated by the given angle
    """
    height, width = image.shape
    x_shift = (width / 2.) - centre[0]
    y_shift = (height / 2.) - centre[1]
    M = np.array([[1, 0, x_shift], [0, 1, y_shift]])
    centred = cv2.warpAffine(image, M, (width, height), borderValue=fill_outside)
    R = cv2.getRotationMatrix2D((width / 2, height / 2), np.degrees(angle), 1)
    stabilised = cv2.warpAffine(centred, R, (width, height), borderValue=fill_outside)
    return stabilised


def contour_info(contour):
    """Uses image moments to find the centre and orientation of a contour

    Parameters
    ----------
    contour : array like
        A contour represented as an array

    Returns
    -------
    c, theta : array, float
        The centre of the contour and its orientation (radians, -pi/2 < theta <= pi/2)
    """
    moments = cv2.moments(contour)
    try:
        c = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    except ZeroDivisionError:
        c = np.mean(contour, axis=0)
        c = tuple(c.squeeze())
    theta = 0.5 * np.arctan2(2 * moments["nu11"], (moments["nu20"] - moments["nu02"]))
    return c, theta
