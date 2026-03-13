'''Based on Daniel's code'''
import cv2
import numpy as np
from libimage import RPiCamV3_img_shape

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def unsharp_mask(
        image,
        kernel_size=(5, 5),
        sigma=1.0,
        amount=1.0,
        threshold=0):
    """Return a sharpened version of the image, using an unsharp mask.

    https://en.wikipedia.org/wiki/Unsharp_masking
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm"""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        # OpenCV4 function copyTo
        np.copyTo(sharpened, image, where=low_contrast_mask)
    return sharpened

def beautify_frame(img):
    """Undistort, sharpen, hist-equalize and label image."""
    img = unsharp_mask(img, amount=1.5)

    # Histogram equalization
    img = cv2.equalizeHist(img)
    #img = cv2.GaussianBlur(img, (3, 3), 0)

    return img

def beautify_frame_graz(img, rpi:int):
    """Undistort, sharpen, hist-equalize and label image."""
    # First cut the image to the relevant part (remove the black borders)
    if rpi == 1:
        img = rotate_image(img, -2)
        img = img[950:2600, 270:3300]
    elif rpi == 2:
        img = rotate_image(img, 1.5)
        img = img[600:2400, 330:3500]
    elif rpi == 3:
        img = rotate_image(img, -0.5)
        img = img[700:2350, 650:3700]
    elif rpi == 4:
        img = rotate_image(img, 0)
        img = img[860:2550, 400:3360]

    # Then scale the image such that the final size is RPiCamV3_img_shape
    img = cv2.resize(img, (RPiCamV3_img_shape[1], RPiCamV3_img_shape[0]))
    
    img = beautify_frame(img)

    return img
