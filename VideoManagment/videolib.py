import cv2, os
from tqdm import tqdm
import pandas as pd
import numpy as np
from libimage import RPiCamV3_img_shape, RPiCamV3_img_shape_RGB


def fig_to_rgb_array(fig, rgb=False):
    '''
    Converts a Matplotlib figure to a 3D NumPy array (height, width, 3) of RGB values.
    Parameters:
    - fig: Matplotlib figure object to be converted.
    - rgb: bool, if True, returns RGB format, else BGR format (for OpenCV compatibility).
    Returns:
    - buf: 3D NumPy array of shape (height, width, 3) containing RGB/BGR values.
    '''
    # Render the figure
    fig.canvas.draw()

    # Get the renderer (this is where pixel data lives)
    renderer = fig.canvas.get_renderer()

    # Extract RGB buffer as a NumPy array
    buf = np.asarray(renderer.buffer_rgba())[:, :, :3]  # drop alpha channel
    if not rgb:
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    return buf

def cropFrameToContent(frame: np.ndarray, padding: int = 0) -> np.ndarray:
    '''
    Crops the given frame (RGB or BGR) to the content area, with the specified padding.
    The function assumes that the contour is white (255, 255, 255) around the content area.
    Parameters:
    - frame: 3D NumPy array representing the image/frame to be cropped.
    - padding: int, number of pixels to add as padding around the content area.
    Returns:
    - cropped_frame: 3D NumPy array of the cropped image/frame.
    '''
    # Check that the frame is a 3D array
    assert frame.ndim == 3 and frame.shape[2] == 3, "frame must be RGB/BGR"
    assert padding >= 0, "padding must be non-negative"

    content = np.any(frame < 250, axis=2)
    y, x = np.where(content)

    if y.size == 0 or x.size == 0:
        # no content found
        return frame

    miny, maxy = max(0, y.min() - padding), min(frame.shape[0], y.max() + padding)
    minx, maxx = max(0, x.min() - padding), min(frame.shape[1], x.max() + padding)
    return frame[miny:maxy, minx:maxx]

def generateVideoFromDir():
    '''
    This function generates a video from a sequence of pictures in a directory.
    '''
    pass

def generateVideoFromList(imgs:list, dest, name:str="video", fps:int=10, grayscale:bool=True):
    '''
    This function generates a video from a list of images.
    '''
    # Checks on the inputs
    if not os.path.isdir(dest):
        os.makedirs(dest)
    if len(imgs) == 0:
        raise ValueError("imgs must be a non-empty list")

    # Generate mp4 video from final_imgs
    if grayscale:
        height, width = imgs[0].shape
    else:
        height, width, _ = imgs[0].shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    # Convert destination from PosixPath to string
    name = str(dest)+'/'+name+".mp4"

    # If file exists, add a number to the name
    base_name, ext = os.path.splitext(name)
    counter = 1
    while os.path.isfile(name):
        name = f"{base_name}_{counter}{ext}"
        counter += 1
    
    video = cv2.VideoWriter(name, fourcc, fps, size, isColor=not grayscale)

    # Iterate over the frames and write each one to the video
    for frame in tqdm(imgs, desc="Writing video", unit="frame"):
        video.write(frame)

    # Release the VideoWriter object
    video.release()

def initVideoWriter(dest, shape, name:str="video",fps:int=10):
    '''
    This function initializes a VideoWriter object to write frames to a video file.
    If the video file already exists, it will be overwritten.
    
    :param dest: str, destination directory to save the video.
    :param shape: tuple, shape of the frames to be written to the video (height, width) or (height, width, channels).
    :param name: str, name of the video file (without extension).
    :param fps: int, frames per second for the video.
    '''
    assert len(shape) in [2, 3], "shape must be a tuple of length 2 or 3"
    grayscale = len(shape) == 2
    # Checks on the inputs
    if not os.path.isdir(dest):
        raise ValueError("dest must be a valid directory")
    
    # Generate mp4 video from final_imgs
    if grayscale:
        height, width = shape
    else:
        height, width, _ = shape
    size = (width, height)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    # Convert destination from PosixPath to string
    name = str(dest)+'/'+name+".mp4"
    video = cv2.VideoWriter(name, fourcc, fps, size, isColor = not grayscale)
    return video

def imageHiveOverview(imgs: list, rgb: bool = False, img_names: list[str]= None, dt: pd.Timestamp = None, use_cet_time:bool = False, valid: bool = True):
    '''
    Generates a global image with the 4 images of the hives. If provided, also adds the img_names on the pictures.
    Parameters:
    - imgs: list of 4 images (numpy arrays) to be concatenated.
    - rgb: bool, if True, the input images are in RGB format, else BGR format.
    - img_names: list of str, optional, names of the images to be put on the images.
    - dt: optional, datetime to be displayed on the final image.
    - use_cet_time: bool, if True, convert dt to CET before displaying.
    - valid: bool, if False, the overview will be visually marked as invalid.
    '''
    if img_names is not None:
        # put the img names on each image
        for j in range(len(imgs)):
            cv2.putText(imgs[j], img_names[j], (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    # Transform Nones in imgs into black images
    for i in range(len(imgs)):
        if imgs[i] is None:
            imgs[i] = np.zeros(RPiCamV3_img_shape_RGB if rgb else RPiCamV3_img_shape, dtype=np.uint8)

    # Concatenate the images horizontally
    img_top = cv2.hconcat([imgs[0], imgs[2]]) # Frame 1 and 3 on top
    img_bottom = cv2.hconcat([imgs[1], imgs[3]]) # Frame 2 and 4 on bottom
    img = cv2.vconcat([img_top, img_bottom])
    # Resize the image to 4K
    img = cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_LINEAR)

    if dt is not None:
        # Make sure it is tz-aware
        assert dt.tzinfo is not None, "dt must be tz-aware"
        if use_cet_time:
            dt = dt.tz_convert('Europe/Zurich').strftime("%d/%m/%Y-%H:%M")
        else:
            dt = dt.tz_convert('UTC').strftime("%y%m%d-%H%M") + "Z"  # Convert to UTC and format as string
        # Write the timestamp in black ontop of the white rectangle
        (text_width, text_height), _ = cv2.getTextSize(dt, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        rectangle_bgr = (255, 255, 255)
        box_coords = ((1700, 1060 + 15), (1700 + text_width, 1060 - text_height - 15))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, dt, (1700, 1060), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    
    if not valid:
        # Convert img to RGB or BGR if it is grayscale
        if len(img.shape) == 2:  # Grayscale image
            if rgb:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # If the image is not valid, put a transparent filter on the image
        overlay = img.copy()
        red_color = (255, 0, 0) if rgb else (0, 0, 255)
        cv2.rectangle(overlay, (0, 0), (3840, 2160), red_color, -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        # Add "Invalid" text on the image
        cv2.putText(img, "Invalid dt", (1500, 500), cv2.FONT_HERSHEY_SIMPLEX, 6, red_color, 15, cv2.LINE_AA)
        cv2.putText(img, "Invalid dt", (1500, 1600), cv2.FONT_HERSHEY_SIMPLEX, 6, red_color, 15, cv2.LINE_AA)

    return img