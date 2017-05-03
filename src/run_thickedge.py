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
from argparse import ArgumentParser, Action, ArgumentError

# Vision Imports
import cv2

# Local Imports
from polygonal_approximation import thick_polygonal_approximate
from background_subtract import init_bgs, process_frame

#-------------------------------------------------------------------------------
# Internal Definitions
#-------------------------------------------------------------------------------

# The default camera resolution. This isn't actually a valid resolution, but
# when this is passed to the `VideoCapture.set` function, it will cause the
# camera to use its maximum resolution.
DEFAULT_RESOLUTION = (10000, 10000)

# The default thickness to use. When the thickness is between 0 and 1, it is
# treated as a fraction of the smaller image dimension (typically the height).
# In this case, this is 1% of that dimension.
DEFAULT_THICKNESS = 0.01

#-------------------------------------------------------------------------------
# Command-Line Parsing and Sanity Checks
#-------------------------------------------------------------------------------

class ParseResolution(Action):
    """An argparse action subclass for parsing image resolutions of the form
    <int>x<int> on the command line."""

    def __init__(self, option_strings, dest, **kwargs):
        """Initialization method for the action. Simply invokes the super
        method."""

        super(ParseResolution, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """The action method for the class. This parses the resolution values
        from the given command line string, placing it in the namespace."""

        # Parse either one resolution or a list of them, and set the associated
        # destination attribute in the namespace.
        if self.nargs is None:
            resolution = self._parse_resolution(values)
        else:
            resolution = [self._parse_resolution(string) for string in values]
        setattr(namespace, self.dest, resolution)

    def _parse_resolution(self, string):
        """Parses the given string as a resolution (widthxheight), returning a
        (width, height) tuple."""

        split = string.lower().split('x')
        if len(split) != 2:
            raise ArgumentError(self, "invalid image resolution value: "
                    "'{}'".format(string))
        try:
            return (int(split[0]), int(split[1]))
        except ValueError:
            raise ArgumentError(self, "invalid image resolution value: "
                    "'{}'".format(string))

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
            default=DEFAULT_THICKNESS, help="The thickness parameter to use "
            "for the thick-edge polygonal approximation algorithm. If this "
            "value is between 0 and 1, then it is used as a fraction of the "
            "smallest image dimension. Otherwise, it is treated as a distance.")
    parser.add_argument("-r", "--camera-resolution", dest="camera_resolution",
            action=ParseResolution, default=DEFAULT_RESOLUTION,
            metavar="WIDTHxHEIGHT", help="The resolution to use for the "
            "specified camera device. The nearest available resolution will be "
            "used. By default, the maximum resolution is used.")
    parser.add_argument("-s", "--show-intermediate", dest="show_intermediate",
            action="store_true", help="Display the intermediate steps of "
            "processing on the video frame, along with the original display.")

    # Get the camera resolution, and move it to different names in the namespace
    args = parser.parse_args()
    (width, height) = args.camera_resolution
    args.camera_width = width
    args.camera_height = height

    return args

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
    elif args.camera_width <= 0 or args.camera_height <= 0:
        msg = "Error: Image resolution must be a positive number."
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

    # If a camera is being used, set it to its maximum resolution, or the one
    # specified by the user.
    if args.video_file is None:
        video_stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, args.camera_width)
        video_stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, args.camera_height)

    # If the thickness was specified as a fraction compute the thickness as the
    # fraction of the smaller image dimension.
    frame_width = int(round(video_stream.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(video_stream.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    if args.thickness < 1.0:
        thickness = args.thickness * min(frame_width, frame_height)
    else:
        thickness = args.thickness

    # Inform the user of the camera resolution and the controls
    print("\nThick Polygonal Approximation Application:")
    print("\tVideo Stream Resolution: {:d}x{:d}".format(frame_width,
            frame_height))
    print("\tPress 's' to save the current frame, 'q' to quit.")
    print("\tPress 'r' to reset the background subtraction algorithm's "
            "state.\n")

    try:
        # Initialize the BGS, grab the polygons from the first frame.
        bgs_algorithm = init_bgs()
        (frame, polygons) = process_frame(bgs_algorithm, video_stream,
                show_intermediate=args.show_intermediate)

        # While frames remain, iterate over each and overlay the polygons.
        frame_num = 0
        while frame is not None:
            # Approximate the polygons for the outlines, and overlay them.
            thick_polygons = [thick_polygonal_approximate(polygon, thickness)
                    for polygon in polygons]
            cv2.polylines(frame, thick_polygons, isClosed=True,
                    color=(180, 40, 100), thickness=int(thickness/2))

            # Show the frame with the polygons overlaid, and process user keys.
            cv2.imshow(video_file, frame)
            key_pressed = chr(cv2.waitKey(50) & 0xFF)

            # Take the appropriate action based on the key that was pressed.
            if key_pressed == 'r':
                print("Resetting background subtraction algorithm state...")
                bgs_algorithm = init_bgs()
            elif key_pressed == 's':
                frame_path = "{}_{}.png".format(video_name, frame_num)
                print("Saving frame to '{}'...".format(frame_path))
                cv2.imwrite(frame_path, frame)
            elif key_pressed == 'q':
                print("Quitting...")
                break

            # Read the next frame and extract its polygons
            (frame, polygons) = process_frame(bgs_algorithm, video_stream,
                    show_intermediate=args.show_intermediate)
            frame_num += 1

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Quitting...")

    # Destroy all the open windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
