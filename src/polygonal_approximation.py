#!/usr/bin/env python3

"""
polygonal_approximation.py

Brandon Perez (bmperez)
Sohil Shah (sohils)

Tuesday, April 25, 2017 at 04:28:33 PM EDT

This file contains the implementation of polygonal approximation through
dominant point detection.

The goal of this algorithm is to reduce the set of points needed to define a
polygon to as small as possible of a set that still accurately characterizes the
shape. This is done by only considering the dominant points along the closed
polyline that defines that polygon.

This implementation is based on the paper "A Computer Vision Framework for
Detecting Dominant Points on Contour of Image-Object Through Thick-Edge
Polygonal Approximation" by Saha, et al.

This file's interface can be imported as follows:
    from polygonal_approximation import thick_polygonal_approximate
"""

# General Imports
import numpy

#-------------------------------------------------------------------------------
# Internal Definitions
#-------------------------------------------------------------------------------

#: The default thickness of the edge that is used for the polygonal
#: approximation, in terms of pixels. This is the threshold used to determine
#: when to split a curve recursively, or to classify the endpoints as dominant
#: points. The thickness is the sum distance of the global maximum and minimum
#: along the polyline to the line between the two endpoints.
DEFAULT_THICKNESS = 8.0

#-------------------------------------------------------------------------------
# Public Interface
#-------------------------------------------------------------------------------

def thick_polygonal_approximate(points, thickness=DEFAULT_THICKNESS):
    """Given an array of points defining a closed polyline, or polygon, filters
    the points so that only the dominant points along the closed polyline
    remain.

    Args:
        points (numpy.ndarray): An 2xN matrix of the points that define the
                closed polyline, where N is the number of points. These are
                assumed to be sorted in clockwise or counterclockwise order.
        thickness (float): The threshold used to determine whether a thick
                curve needs to be split further, in terms pixels. This argument
                is optional, and defaults to 8.0.
    Returns:
        numpy.ndarray: An 2xM matrix of the dominant points on the closed
                polyline, where M <= N. This preserves the ordering of points.
    """

    assert(points.ndim == 2 and points.shape[0] == 2 and points.shape[1] >= 2)
    assert(thickness > 0.0)
    assert(numpy.unique(points).size == points.size)

    return _polyline_approx(points, thickness)

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

def _polyline_approx(points, thickness_threshold):
    """Approximates the given points using a thick-polyline approximation.

    Given a polyline defined by the given sorted points, and a thickness that
    defines a dominant curve, this function finds the approximation of the
    polyline containing only dominant points.
    """

    assert(points.shape[1] >= 2)

    # Compute the linear regression line for the polyline's points.
    (_, num_points) = points.shape
    (xs, ys) = (points[0, :], points[1, :])
    coefficient_matrix = numpy.hstack([xs.T, numpy.ones(num_points, 1)])
    (solution, _, _, _) = numpy.linalg.lstsq(coefficient_matrix, ys)
    (slope, y_intercept) = solution

    # Convert the line to standard form, and normalize the coefficients for a
    # unit line vector.
    line_vector_unnorm = numpy.array([-slope, 1, y_intercept])
    line_vector = numpy.linalg.norm(line_vector_unnorm)

    # Create a mask to partition points based on whether they are above or below
    # the line from their distance to the line.
    points_hom = numpy.vstack((points, numpy.ones(1, num_points)))
    point_distances = numpy.dot(line_vector, points_hom)
    below_mask = point_distances < 0

    # Find the extreme points in the above and below sets, and use them to
    # compute the thickness of the polyline based on the distances.
    below_extremum_index = _find_extremum(points, point_distances, below_mask)
    above_extremum_index = _find_extremum(points, point_distances, ~below_mask)
    thickness = _get_array_value(point_distances, below_extremum_index, 0.0)
    thickness += _get_array_value(point_distances, above_extremum_index, 0.0)

    # If the thickness is below the threshold, then the two endpoints are
    # dominant points, so we're done. Otherwise, partition the points.
    if thickness < thickness_threshold:
        return points[:, (0, -1)]

    # Otherwise, partition the points based on the extreme values. Recursively
    # approximate the smaller polylines represented by these points.
    (points1, points2, points3) = _partition_points(points,
            below_extremum_index, above_extremum_index)
    dominant_points = numpy.empty()
    dominant_points1 = _polyline_approx(points1, thickness_threshold)
    dominant_points = numpy.hstack((dominant_points, dominant_points1))
    dominant_points2 = _polyline_approx(points2, thickness_threshold)
    dominant_points = numpy.hstack((dominant_points, dominant_points2))
    if points3 is not None:
        dominant_points3 = _polyline_approx(points3, thickness_threshold)
        dominant_points = numpy.hstack((dominant_points, dominant_points3))

    return dominant_points

def _find_extremum(points, point_distances, subset_mask):
    """Finds the index of the minimum or maximum point from the given subset of
    the points."""

    # If the point set is empty, then no extremum exists.
    points_subset = points[subset_mask]
    point_distances_subset = point_distances[subset_mask]
    if points_subset.size == 0:
        return None

    # Otherwise, find the extreme value based on the absolute distance to the
    # line within the specified subset.
    extremum_subset_index = numpy.argmax(numpy.absolute(point_distances_subset))
    extremum = points_subset[:, extremum_subset_index]
    return _find_column(points, extremum)

def _find_column(array, column):
    """Finds the column in the given matrix, returning its index. Fails if the
    column does not exist, or is present more than once."""

    # Search for the column in the array, making sure the input column is a
    # column vector.
    column_vector = column.reshape((len(column), 1))
    element_mask = (array == column_vector).all(axis=0)
    column_indices = numpy.where(element_mask)

    assert(column_indices.size == 1)
    return column_indices[0]

def _get_array_value(array, index, default):
    """Returns the array[index], or default if the index is None"""

    if index is not None:
        return array[index]
    else:
        return default

def _partition_points(points, below_extremum_index, above_extremum_index):
    """Partitions the points into 2 or 3 sets, which span from the starting
    point to each of the extrema, and finally to the end point. If a extremum
    does not exist, then the third partition is None."""

    (_, num_points) = points.shape
    assert(below_extremum_index is not None or above_extremum_index is not None)
    assert(below_extremum_index is None or
            below_extremum_index + 1 < num_points)
    assert(above_extremum_index is None or
            above_extremum_index + 1 < num_points)
    assert(above_extremum_index != below_extremum_index)

    # Partition the points based on which extrema doesn't exist. If both do,
    # then partition, making sure to respect the ordering.
    if below_extremum_index is None:
        return (
            points[:, 0:above_extremum_index],
            points[:, above_extremum_index+1:],
            None,
        )
    elif below_extremum_index is None:
        return (
            points[:, 0:below_extremum_index],
            points[:, below_extremum_index+1:],
            None,
        )
    elif below_extremum_index < above_extremum_index:
        return (
            points[:, 0:below_extremum_index],
            points[:, below_extremum_index+1:above_extremum_index],
            points[:, above_extremum_index+1:]
        )
