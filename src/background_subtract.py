"""
background_subtract.py

Brandon Perez (bmperez)
Sohil Shah (sohils)

This file contains the implementation of the video pre-processing
pipeline.

This file contains functions to process either a video from a webcam
or a file. The processing steps are as follows:
  - Use chosen background subtraction algorithm to get a forground
  - Erode the image to remove noise
  - Dilate the image to reduce holes in the moving object
  - Use OpenCV edge detection to detect an outline of the binary 
    image
  - Find connected components
  - Throw out outlines that are too small from noise
  - Find points in order for polygon for each connected component
  - Output as list of list of points

The background subtraction library used here can be found at:
https://github.com/andrewssobral/bgslibrary

Requirements: 
  - libbgs requires that there is a folder named 'config' in the directory 
    it is run from
  - Requires python2
  - Requires OpenCV 2

"""


import numpy as np
import cv2
import libbgs

# Choose here which algorithm to use for background subtraction
# A list of algorithms is available at: 
# https://github.com/andrewssobral/bgslibrary/wiki/List-of-available-algorithms
bgs = libbgs.LBMixtureOfGaussians() #libbgs.FrameDifference()

# Display intermediate images for debugging
debug = 1


#------------------------------------------------------------------------------
# Public Interface
#------------------------------------------------------------------------------
def process_camera(camera_dev):
    """
    Reads frames from Camera one at a time as they are ready and
    preprocesses them for polygons. 

    Args:
        camera_dev (string): Filename of the camera device

    Returns:
        list of numpy array lists: Each list represents one frame,
                                   contains list of numpy array of
                                   points. 
                                   Eg. frames[0][0] is the first 
                                   polygon of the first frame and
                                   is a numpy array of points.

    """
    frames = []

    capture = cv2.VideoCapture(camera_dev)
    while not capture.isOpened():
        capture = cv2.VideoCapture(camera_dev)
        cv2.waitKey(1000)
        print "Waiting for Camera"

    pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = capture.read()

        if flag:
            pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

            fg_edges = _process_frame(frame)
            polys = find_polys(fg_edges)

            frames.append(polys)

        else:
            capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print "Frame is not ready"
            cv2.waitKey(1000)

        if cv2.waitKey(10) > 0:
            break

    if debug:
        cv2.destroyAllWindows()

    return frames

def process_video(video_file):
    """
    Reads frames from a video file of any format and returns a list
    of polygons for every frame. 

    Args:
        video_file (string): Filename of video file

    Returns:
        list of numpy array lists: Each list represents one frame,
                                   contains list of numpy array of
                                   points. 
                                   Eg. frames[0][0] is the first 
                                   polygon of the first frame and
                                   is a numpy array of points.

    """
    frames = []

    capture = cv2.VideoCapture(video_file)
    if not capture.isOpened():
        print("Could not open file: ", video_file)

    pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = capture.read()
        pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

        fg_edges = _process_frame(frame)
        polys = _find_polys(fg_edges)

        frames.append(polys)

        # End of video
        if capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break

    if debug:
        cv2.destroyAllWindows()

    return frames


#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------
def _process_frame(frame):    
    """
    Using a frame, preprocesses with erode and dilate to remove noise and holes.
    Then, uses Canny edge detector to detect polygonal edge.

    Args:
        frame (numpy array): frame image to preprocess

    Returns:
        edges (numpy array): binary image of edges
    """
    img_output = bgs.apply(frame)

    # Post process video for clearer mask
    # None for kernel uses default 3x3 gaussian kernel
    img_post = cv2.erode(img_output, None)
    img_post = cv2.dilate(img_post, None, iterations=10)
    
    # Canny edge detector to find all objects
    edges = cv2.Canny(img_post, 2000, 4000, apertureSize=5)


    if debug:
        cv2.imshow('video', frame)
        cv2.imshow('img_output', img_output)
        cv2.imshow('img_mask', img_post)
        cv2.imshow('edges', edges)

        # Pause 50ms between frames
        cv2.waitKey(50)

    return edges

def _find_polys(edges):
    """
    Using the edges image, finds polygons and represents as ordered list of points.

    Args:
        edges (numpy array): binary image of edges

    Returns:
        contours (list of numpy arrays): list of polygons as arrays of points
    """

    # Turns out this just does everything we need
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours
    

