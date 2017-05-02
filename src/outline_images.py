#! /usr/bin/env python2

"""
outline_images.py

Brandon Perez (bmperez)
Sohil Shah (sohils)

This file contains a script for running polygonal approximation on a series of
binary images.

The script expects a directory of binary images that contain a single shape. The
script goes through images one at a time, traces the polygon around the single
shape, then runs polygonal approximation. The image with polygon overlaid and
another with the image with the approximated polygon are then saved to the
output directory. In addition, statistics for each image are saved to a text
file.

The polygon image will have the name `filename_polygon.ext`, while the
approximated polygon image will have the name `filename_approx.ext`. At the top
level of the output directory will be a `statistics.txt` file, containing the
stats for each image.

An example dataset that has this format is the MPEG-7 Shape Test Set:
    http://www.dabi.temple.edu/~shape/MPEG7/dataset.html
"""

# General Imports
from os import walk, path, access, makedirs, R_OK
from argparse import ArgumentParser
import csv

# Vision Imports
import numpy
import cv2
from matplotlib import pyplot

# Local Imports
from polygonal_approximation import thick_polygonal_approximate
from polygonal_approximation import compare_polygons

#-------------------------------------------------------------------------------
# Internal Parameters
#-------------------------------------------------------------------------------

# The default thickness used for the thick-edge polygonal approximation
# algorithm. This is a percentage of the minimum image dimension (max of width
# and height).
DEFAULT_THICKNESS = 0.03

#-------------------------------------------------------------------------------
# Command-Line Parsing and Sanity Checks
#-------------------------------------------------------------------------------

def parse_arguments():
    """Parses the arguments specified by the user on the command line."""

    # Setup the command line options
    parser = ArgumentParser(description="Runs thick-edge polygonal "
            "on the specified set of binary shape images and saves the "
            "the image with the polygon overlaid and another with the "
            "approximated polygon overlaid to file.")
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

    return parser.parse_args()

def sanity_check(args):
    """Runs a basic sanity check on the arguments specified by the user."""

    # Template for the error message
    msg_template = "Error: {}: {}"

    # Check that the thickness, data directory, and output directory are valid.
    if args.thickness <= 0 or args.thickness >= 1.0:
        msg = "Error: Thickness must be positive and less than 1.0"
    elif not path.exists(args.data_dir):
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

def overlay_polygon(image, polygon, thickness, image_name, polygon_name,
        output_dir):
    """Overlays the given polygon on the image, plotting the line and
    potentially the points. This is then saved the specified output
    directory. The figure is returned."""

    # Force the polygon to be closed
    polygon_closed = numpy.append(polygon, polygon[0:1, :], axis=0)

    # Plot the polygon over the image
    figure = pyplot.figure()
    pyplot.imshow(image)
    if polygon_name == "polygon":
        pyplot.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'r', color='r')
    else:
        pyplot.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'r', color='r',
                linewidth=thickness)
        pyplot.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'o', color='b',
                markersize=thickness+3)

    # Trim all of the unnecessary whitespace from the plot
    pyplot.axis('off')
    figure.axes[0].xaxis.set_major_locator(pyplot.NullLocator())
    figure.axes[0].yaxis.set_major_locator(pyplot.NullLocator())

    # Save the figure to the specified output location, then return the figure
    (name, extension) = path.splitext(image_name)
    polygon_file = "{}_{}{}".format(name, polygon_name, extension)
    polygon_path = path.join(output_dir, polygon_file)
    pyplot.savefig(polygon_path, bbox_inches='tight', pad_inches=0)

    return figure

def approximate_polygon(args, dir_path, image_name, image):
    """Traces the polygon of the shape in the image, then approximates it with
    thick-edge approximation. The two polygons are overlaid on the original
    image and saved to file. Stats the polygons are returned."""

    # Find the polygons in the image, and verify there is only one.
    gray_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    (contours, _) = cv2.findContours(gray_image, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    (num_points, _, num_dim) = contours[0].shape
    polygon = numpy.reshape(contours[0], [num_points, num_dim])

    # Calculate the thickness parameter and approximate the polygon.
    (height, width) = gray_image.shape
    thickness = min(height, width) * args.thickness
    approx_polygon = thick_polygonal_approximate(polygon, thickness)

    # Create the subdirectory in the output directory if it doesn't exist
    subdir_path = path.relpath(dir_path, args.data_dir)
    output_subdir_path = path.join(args.output_dir, subdir_path)
    if not path.exists(output_subdir_path):
        makedirs(output_subdir_path)

    # Compare the polygons represented by the two polygons
    (vertices, _) = polygon.shape
    (approx_vertices, _) = approx_polygon.shape
    (area, approx_area, vertex_diff, area_diff) = compare_polygons(polygon,
            approx_polygon)

    # Report the statistics to the user
    image_path = path.join(subdir_path, image_name)
    print("\nImage {} Statistics:".format(image_path))
    print("\tPolygon Vertices:              {:<10}".format(vertices))
    print("\tPolygon Area:                  {:<10.3f}".format(area))
    print("\tApproximated Polygon Vertices: {:<10} ({:0.3f}%)".format(
            approx_vertices, vertex_diff * 100))
    print("\tApproximated Polygon Area:     {:<10.3f} ({:0.3f}%)".format(
            approx_area, area_diff * 100))

    # Create figures for and save the image with the original and approximated
    # polygons overlaid.
    polygon_figure = overlay_polygon(image, polygon, thickness, image_name,
            "polygon", args.output_dir)
    approx_figure = overlay_polygon(image, approx_polygon, thickness,
            image_name, "approx", args.output_dir)

    # If specified by user, show each plot, then close the plots
    if args.show_images:
        pyplot.show()
    pyplot.close(polygon_figure)
    pyplot.close(approx_figure)

    return (image_path, vertices, approx_vertices, area, approx_area,
            vertex_diff * 100, area_diff * 100)

def main():
    """The main function for the script."""

    # Parse the arguments, and run a basic sanity check
    args = parse_arguments()
    sanity_check(args)

    # Create the output directory, if it does not exist
    if not path.exists(args.output_dir):
        makedirs(args.output_dir)

    # Iterate over each image in the output directory, and process it
    statistics = list()
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
            stats = approximate_polygon(args, dir_path, file_name, image)
            if stats is None:
                print("Warning: {}: Image does not have precisely one "
                        "shape.".format(image_path))
                continue
            statistics.append(stats)

    # Save the statistics out to file as a csv
    statistics_file_path = path.join(args.output_dir, 'statistics.csv')
    csv_header = ("Image", "Polygon Vertices", "Approximated Vertices",
            "Polygon Area", "Approximated Area", "Vertex Difference (%)",
            "Area Difference (%)")
    with open(statistics_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        csv_writer.writerows(statistics)

    # Report the average vertex and area reduction
    vertex_diffs = [v for (_, _, _, _, _, v, _) in statistics]
    area_diffs = [abs(a) for (_, _, _, _, _, _, a) in statistics]
    print("\n\n--------------------------------------------------------------")
    print("Overall Statistics:")
    print("\tAverage Vertex Reduction: {:0.3f}%".format(
            sum(vertex_diffs) / len(vertex_diffs)))
    print("\tAverage Area Difference: {:0.3f}%".format(
            sum(area_diffs) / len(area_diffs)))

if __name__ == '__main__':
    main()
