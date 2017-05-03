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
    it is run from (that already exists).
  - Requires python2
  - Requires OpenCV2
"""

# Vision Imports
import numpy
import cv2

# Background subtraction library import.
import libbgs

#------------------------------------------------------------------------------
# Public Interface
#------------------------------------------------------------------------------

def init_bgs():
    """Initializes and returns a background subtraction algorithm object.

    To reset the algorithm class's state, invoke this function again, and assign
    it to the same variable. This will automatically deconstruct the previous
    instance of the algorithm.

    Outputs:
        (libbgs.Algorithm): A BGS library algorithm class instance, that
            implements one of the many variants of background subtraction.
    """

    # Change this class to change the background subtraction algorithm The
    # algorithm that was found to work the best was MultiCue background
    # subtraction. The paper for this algorithm can be found at:
    #   https://link.springer.com/chapter/10.1007%2F978-3-642-37431-9_38
    #
    # Some other good algorithms were:
    #   1. Independent Multimodal Background Subtraction
    #   2. Mixture of Gaussians Background Subtraction.
    #
    # A complete list of the BGS library's available algorithms on its Github
    # wiki.
    return libbgs.MultiCue()

def process_frame(bgs_algorithm, video_stream, show_intermediate=False):
    """Reads the next frame from the video stream, detects all the moving
    objects in the frame, and returns polygons representing their outlines.

    If no frames are remaining in the video stream (end of a video file, or a
    camera is disconnected), then this returns None.

    Args:
        bgs_algorithm (libbgs.Algorithm): The BGS library algorithm class to use
            for background subtraction.
        video_stream cv2.VideoCapture): The VideoCapture object representing the
            video stream (either a camera or video file), from which the next
            frame is read.
        show_intermediate (bool): Show the intermediate images after each step
            of processing as OpenCV windows. The user must call `cv2.waitKey`
            for the windows to be displayed. Optional, defaults to False.

    Returns:
        numpy.ndarray: A HxWxD array representing the next frame in the given
                video stream. If no frames remain in the video stream, then None
                is returned.
        list (numpy.ndarray): A list of Nx2 matrices where each matrix defines
                the points of an outline of one of the moving objects in the
                frame. N is the number of points for a given outline. If no
                frames remain in the video stream, then None is returned.
    """

    # Read the next frame from the camera. If no such frame is available, return
    # None.
    (valid, frame) = video_stream.read()
    if not valid:
        return (None, None)

    # Subtract the background from the frame to get the moving objects, and
    # process it to get their edges.
    (bgs_frame, postprocessed_frame, edge_frame) = _find_moving_edges(
            bgs_algorithm, frame)

    # If requested, show the intermediate processing steps as images.
    if show_intermediate:
        cv2.imshow('Video Frame', frame)
        cv2.imshow('Background Subtracted Mask', bgs_frame)
        cv2.imshow('Postprocessed Mask', postprocessed_frame)
        cv2.imshow('Edges', edge_frame)

    # Find the outlines of the moving objects as polygons.
    polygons = _find_polygons(edge_frame)
    return (frame, polygons)

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

def _find_moving_edges(bgs_algorithm, image):
    """Using the given background subtraction algorithm and various
    preprocessing steps, find the edges of moving objects in the image."""

    # Apply background subtraction to get a mask for moving objects.
    bgs_image = bgs_algorithm.apply(image)

    # Process the mask to fill in holes and generally smooth it out, leading to
    # a better result. Note that None indicates to use a 3x3 Gaussian kernel.
    dilated_image = cv2.dilate(bgs_image, kernel=None, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel=None, iterations=3)
    dilated2_image = cv2.dilate(eroded_image, kernel=None, iterations=10)
    postprocessed_image = cv2.erode(dilated2_image, kernel=None, iterations=8)

    # Zero out any edges that are on the border of the image. This prevents the
    # `findCountours` from creating a polygon that traces both sides of edges.
    postprocessed_image[0, :] = 0
    postprocessed_image[-1, :] = 0
    postprocessed_image[:, 0] = 0
    postprocessed_image[:, -1] = 0

    # Use a Canny edge detector to find the edges of all the objects, and smooth
    # it out with a dilation.
    canny_image = cv2.Canny(postprocessed_image, threshold1=1000,
            threshold2=3000, apertureSize=5)
    edge_image = cv2.dilate(canny_image, kernel=None, iterations=1)

    return (bgs_image, postprocessed_image, edge_image)

def _find_polygons(edge_image):
    """Given a binary image of edges, find the polygons representing the
    outlines of all the objects in the image. The polygons are represented as
    ordered list of points (Nx2 matrix)."""

    # Find the contours (outlines) in the image.
    (contours, _) = cv2.findContours(edge_image, mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)

    # Reshape each contour to be an Nx2 image.
    reshaped_contours = [None for i in range(len(contours))]
    for i in range(len(contours)):
        (num_points, _, num_dim) = contours[i].shape
        reshaped_contours[i] = numpy.reshape(contours[i], [num_points, num_dim])
    return reshaped_contours
