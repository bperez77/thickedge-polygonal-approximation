#! /usr/bin/env python2

"""
run_thickedge.py

Brandon Perez (bmperez)
Sohil Shah (sohils)

This file contains the main application for running the polygonal thick-edge
approximation.

This applications allows for the polygonal thick-edge approximation to be run
either on an offline video, or online in real-time on a webcam video. In either
case, the application uses the background subtraction library to get all the
moving objects in each frame. Then, using this, a contour is traced around each
object, the polygonal thick-edge approximation is run, and then this is overlaid
on the video.
"""

# General Imports
from os import path, access, R_OK
from argparse import ArgumentParser

# Vision Imports
import cv2

# Local Imports
from polygonal_approximation import thick_polygonal_approximate
from background_subtract import process_frame

#-------------------------------------------------------------------------------
# Command-Line Parsing and Sanity Checks
#-------------------------------------------------------------------------------

def parse_arguments():
    """Parses the arguments specified by the user on the command line."""

    # Setup the command line options
    parser = ArgumentParser(description="Runs background subtraction, followed "
            "by thick-edge polygonal approximation on all moving objects in "
            "the scene. This can be either run offline on a video file, or "
            "online on a video input from a webcam.")
    parser.add_argument("-f", "--video-file", dest="video_file", type=str,
            required=True, help="The path to the video on which to run the "
            "thick-edge polygonal approximation pipeline.")
    parser.add_argument("-t", "--thickness", dest="thickness", type=float,
            required=True, help="The thickness parameter to use for the "
            "thick-edge polygonal approximation algorithm.")

    return parser.parse_args()

def sanity_check(args):
    """Runs a basic sanity check on the arguments specified by the user."""

    # Template for the error message
    msg_template = "Error: {}: {}"
    msg = ""

    if args.thickness <= 0:
        msg = "Error: Thickness must be positive."
    elif not path.exists(args.video_file):
        msg = msg_template.format(args.video_file, "Video file does not exist.")
    elif not path.isfile(args.video_file):
        msg = msg_template.format(args.video_file, "Video file is not a file.")
    elif not access(args.video_file, R_OK):
        msg = msg_template.format(args.video_file, "Video file lacks read "
                "permissions")
    else:
        return

    print(msg)
    exit(1)

#-------------------------------------------------------------------------------
# Main Application
#-------------------------------------------------------------------------------

def main():
    """The main function for the script."""

    # Parse the arguments, and run a basic sanity check on them.
    args = parse_arguments()
    sanity_check(args)

    # Open up the video file, for displaying the polygons overlaid on frames.
    capture = cv2.VideoCapture(args.video_file)
    if not capture.isOpened():
        print("Error: {}: Unable to open video file.".format(args.video_file))
        exit(1)

    # Extract the basename of the video wtihout its extension
    (video_name, _) = path.splitext(args.video_file)

    # Iterate over each frame in the video, and overlay the polygons on each.
    num_frames = int(round(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
    for i in range(num_frames):
        # Read the next frame, extract the polygons, approximate them with
        # thick-edge, and then overlay them on the video frame.
        (frame, polygons) = process_frame(capture, show_intermediate=False)
        thick_polygons = [thick_polygonal_approximate(polygon, args.thickness)
                for polygon in polygons]
        cv2.polylines(frame, thick_polygons, isClosed=True,
                color=(180, 40, 100), thickness=int(round(args.thickness/2)))

        # Show the frame with the polygons overlaid. If the user presses any
        # key, save the current frame to file.
        cv2.imshow(args.video_file, frame)
        if cv2.waitKey(50) > 0:
            print("Saving file...")
            cv2.imwrite("{}_{}.png".format(video_name, i), frame)

    # Destroy all the open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
