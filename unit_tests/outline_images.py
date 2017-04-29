#!/usr/bin/env python3

"""
outline_images.py

Brandon Perez (bmperez)
Sohil Shah (sohils)

Friday, April 28, 2017 at 9:43:45 PM EDT

This file contains a script for running polygonal approximation on a series of
binary images.

The script expects a directory of binary images that contain a single shape. The
script goes through images one at a time, traces the contour around the single
shape, then runs polygonal approximation. The image with contour overlaid and
another with the image with the approximated contour are then saved to the
output directory. In addition, statistics for each image are saved to a text
file.

The contour image will have the name `filename_contour.ext`, while the
approximated contour image will have the name `filename_approx.ext`. At the top
level of the output directory will be a `statistics.txt` file, containing the
stats for each image.

An example dataset that has this format is the MPEG-7 Shape Test Set:
    http://www.dabi.temple.edu/~shape/MPEG7/dataset.html
"""

# General Imports
import sys
from os import walk, path, access, makedirs, R_OK
from argparse import ArgumentParser

# Vision Imports
import numpy
import cv2
from matplotlib import pyplot

# Add the src directory to the PYTHONPATH
sys.path.append(path.realpath(path.join(path.dirname(__file__), '..', 'src')))

# Local Imports
from polygonal_approximation import thick_polygonal_approximate

#-------------------------------------------------------------------------------
# Internal Parameters
#-------------------------------------------------------------------------------

# The default thickness used for the thick-edge polygonal approximation
# algorithm. This is a percentage of the minimum image dimension (max of width
# and height).
DEFAULT_THICKNESS = 0.01

#-------------------------------------------------------------------------------
# Command-Line Parsing and Sanity Checks
#-------------------------------------------------------------------------------

def parse_arguments():
    """Parses the arguments specified by the user on the command line."""

    # Setup the command line options
    parser = ArgumentParser(description="Runs thick-edge polygonal "
            "on the specified set of binary shape images and saves the "
            "the image with the contour overlaid and another with the "
            "approximated contour overlaid to file.")
    parser.add_argument("-d", "--data-dir", dest="data_dir", required=True,
            help="The path to the directory containing the data to process.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", required=True,
            help="The path to the directory to store the results of the "
            "polygonal approximation. The file names will mirror the ones in "
            "the data directory.")
    parser.add_argument("-t", "--thickness", dest="thickness", type=float,
            default=DEFAULT_THICKNESS, help="The thickness threshold parameter "
            "used for the polygon approximation. This is specified as a "
            "fraction of the minimum image dimension.")
    parser.add_argument("-s", "--show-images", dest="show_images",
            action='store_true', help="After processing each image, show the "
            "image and pause the script. The script continues when the window "
            "is closed.")

    # Parse the arguments, check the thickness is valid
    args = parser.parse_args()
    if args.thickness >= 1.0:
        print("Error: Thickness must be less than 1.0.")
        exit(1)

    return args

def sanity_check(args):
    """Runs a basic sanity check on the arguments specified by the user."""

    # Template for the error message
    msg_template = "Error: {}: {}"
    msg = ""

    if not path.exists(args.data_dir):
        msg = msg_template.format(args.data_dir, "Data directory does not "
                "exist.")
    elif not path.isdir(args.data_dir):
        msg = msg_template.format(args.data_dir, "Data directory is not a "
                "directory.")
    elif not access(args.data_dir, R_OK):
        msg = msg_template.format(args.data_dir, "Data directory lacks read "
                "permissions.")
    elif path.exists(args.output_dir) and not path.isdir(args.output_dir):
        msg = msg_template.format(args.output_dir, "Output directory exists, "
                "but it is not a directory.")
    else:
        return

    print(msg)
    exit(1)

#-------------------------------------------------------------------------------
# Main Application
#-------------------------------------------------------------------------------

def overlay_contour(image, contour, approx):
    """Overlays the given contour on the image, plotting its points and the line
    representing it."""

    # Force the contour to be closed
    contour_closed = numpy.append(contour, contour[0:1, :], axis=0)

    pyplot.imshow(image)
    pyplot.plot(contour_closed[:, 0], contour_closed[:, 1], 'r', color='r')
    if approx:
        pyplot.plot(contour_closed[:, 0], contour_closed[:, 1], 'o', color='b',
                markersize=5)
    pyplot.axis('off')

def approximate_contour(args, dir_path, image_name, image):
    """Traces the contour of the shape in the image, then approximates it with
    thick-edge approximation. The two contours are overlaid on the original
    image and saved to file"""

    # Find the contours in the image, and verify there is only one.
    # FIXME: FindContours should only return two parameters
    gray_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    (_, contours, _) = cv2.findContours(gray_image, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    (num_points, _, num_dim) = contours[0].shape
    contour = numpy.reshape(contours[0], [num_points, num_dim])

    # Calculate the thickness parameter and approximate the contour.
    (height, width) = gray_image.shape
    thickness = min(height, width) * args.thickness
    approx_contour = thick_polygonal_approximate(contour.T, thickness).T

    # Create the subdirectory in the output directory if it doesn't exist
    subdir_path = path.relpath(dir_path, args.data_dir)
    output_subdir_path = path.join(args.output_dir, subdir_path)
    if not path.exists(output_subdir_path):
        makedirs(output_subdir_path)

    # Save the image with the original contour overlaid
    (name, extension) = path.splitext(image_name)
    contour_name = "{}_{}{}".format(name, "contour", extension)
    pyplot.figure(0)
    overlay_contour(image, contour, False)
    pyplot.savefig(path.join(output_subdir_path, contour_name),
            bbox_inches='tight')

    # Save the image with the approximated contour overlaid
    approx_name = "{}_{}{}".format(name, "approx", extension)
    pyplot.figure(1)
    overlay_contour(image, approx_contour, True)
    pyplot.savefig(path.join(output_subdir_path, approx_name),
            bbox_inches='tight')

    # If specified by user, show each plot, then close the plots
    if args.show_images:
        pyplot.show()
    pyplot.close(0)
    pyplot.close(1)

def main():
    """The main function for the script."""

    # Parse the arguments, and run a basic sanity check
    args = parse_arguments()
    sanity_check(args)

    # Create the output directory, if it does not exist
    if not path.exists(args.output_dir):
        makedirs(args.output_dir)

    # Iterate over each image in the output directory, and process it
    #statistics = list()
    for (dir_path, _, file_names) in walk(args.data_dir):
        for file_name in file_names:
            # Attempt to read the file as an image
            image_path = path.join(dir_path, file_name)
            image = cv2.imread(image_path)
            if image is None:
                print("Warning: {}: File is not a valid image file. "
                        "Skipping it.".format(image_path))
                continue

            # Process the image, and save its results to file
            print("Processing {}...".format(image_path))
            approximate_contour(args, dir_path, file_name, image)

if __name__ == '__main__':
    main()
