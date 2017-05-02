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
            help="The path to the video on which to run the thick-edge "
            "polygonal approximation pipeline.")
    parser.add_argument("-d", "--camera-device-id", dest="camera_id", type=int,
            help="The ID number for the camera device on which to run the "
            "thick-edge polygonal approximation pipeline in real time. "
            "Typically, this is 0, as is the case when only one camera is "
            "connected to your computer.")
    parser.add_argument("-t", "--thickness", dest="thickness", type=float,
            required=True, help="The thickness parameter to use for the "
            "thick-edge polygonal approximation algorithm.")
    parser.add_argument("-s", "--show-intermediate", dest="show_intermediate",
            action="store_true", help="Display the intermediate steps of "
            "processing on the video frame, along with the original display.")

    return parser.parse_args()

def sanity_check(args):
    """Runs a basic sanity check on the arguments specified by the user."""

    # Template for the error message
    msg_template = "Error: {}: {}"

    # Check that the arguments were specified correctly, and that either the
    # video file or camera id exists.
    if (args.video_file is None) and (args.camera_id is None):
        msg = ("Error: One of -f/--video-file and -d/--camera-device-id must "
                "be specified.")
    elif (args.video_file is not None) and (args.camera_id is not None):
        msg = ("Error: Only one of -f/--video-file and -d/--camera-device-id "
                "can be specified.")
    elif args.thickness <= 0:
        msg = "Error: Thickness must be positive."
    elif (args.video_file is not None) and (not path.exists(args.video_file)):
        msg = msg_template.format(args.video_file, "Video file does not exist.")
    elif (args.video_file is not None) and (not path.isfile(args.video_file)):
        msg = msg_template.format(args.video_file, "Video file is not a file.")
    elif (args.video_file is not None) and (not access(args.video_file, R_OK)):
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

    # Either open up the video file or the camera specified by the user.
    if args.video_file is not None:
        video_stream = cv2.VideoCapture(args.video_file)
        error_msg = "Error: {}: Unable to open video file.".format(
                args.video_file)
    else:
        video_stream = cv2.VideoCapture(args.camera_id)
        error_msg = "Error: Unable to open camera with id {}.".format(
                args.camera_id)
    if not video_stream.isOpened():
        print(error_msg)
        exit(1)

    # Extract the name of the video. For video files, this is the path without
    # the extension. On Linux systems, we name the cameras as /dev/video[x].
    if args.video_file is not None:
        video_file = args.video_file
        (video_name, _) = path.splitext(args.video_file)
    else:
        video_file = "/dev/video{}".format(args.camera_id)
        video_name = "camera{}".format(args.camera_id)

    # Iterate over each frame in the video, terminating early if the user sends
    # a keyboard interrupt.
    try:
        # Grab the initial frame and polygons from first video stream.
        print("Thick Polygonal Approximation Application:")
        print("Press 's' to save the current frame, 'q' to quit.")
        (frame, polygons) = process_frame(video_stream,
                show_intermediate=args.show_intermediate)

        # While frames remain, iterate over each and overlay the polygons.
        frame_num = 0
        while frame is not None:
            # Approximate the polygons for the outlines, and overlay them.
            thick_polygons = [thick_polygonal_approximate(polygon,
                    args.thickness) for polygon in polygons]
            cv2.polylines(frame, thick_polygons, isClosed=True,
                    color=(180, 40, 100), thickness=int(args.thickness/2))

            # Show the frame with the polygons overlaid. If the user presses
            # 's', save the current frame. Quit if the user presses 'q'.
            cv2.imshow(video_file, frame)
            key_pressed = chr(cv2.waitKey(50) & 0xFF)
            if key_pressed == 's':
                print("Saving file...")
                cv2.imwrite("{}_{}.png".format(video_name, frame_num), frame)
            elif key_pressed == 'q':
                print("Quitting...")
                break

            # Read the next frame and extract its polygons
            (frame, polygons) = process_frame(video_stream,
                    show_intermediate=args.show_intermediate)
            frame_num += 1

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Quitting...")

    # Destroy all the open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
