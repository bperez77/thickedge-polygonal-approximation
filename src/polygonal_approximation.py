"""
polygonal_approximation.py

Brandon Perez (bmperez)
Sohil Shah (sohils)

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
from collections import OrderedDict

# Vision Imports
import numpy

#-------------------------------------------------------------------------------
# Public Interface
#-------------------------------------------------------------------------------

def thick_polygonal_approximate(points, thickness):
    """Given an array of points defining a closed polyline, or polygon, filters
    the points so that only the dominant points along the closed polyline
    remain.

    Args:
        points (numpy.ndarray): An Nx2 matrix of the points that define the
                closed polyline, where N is the number of points. The polygon is
                assumed to be not self-intersecting.
        thickness (float): The threshold used to determine whether a thick
                curve needs to be split further.
    Returns:
        numpy.ndarray: An Mx2 matrix of the dominant points on the closed
                polyline, where M <= N. This preserves the ordering of points.
    """

    assert(points.ndim == 2 and points.shape[1] == 2)
    assert(thickness > 0.0)

    dominant_points_dict = _polyline_approx(points, thickness)
    return numpy.array(list(dominant_points_dict.keys()))

def compare_polygons(points1, points2):
    """Given two arrays of points defined two polygons, or closed polylines,
    this computes the difference in vertices and area between the polygons.

    Args:
        points1 (numpy.ndarray): An Nx2 matrix of points that define the first
                polygon to compare.
        points2 (numpy.ndarray): An Mx2 matrix of points that define the second
                poylgon to compare.
    Returns:
        float: The area of the first polygon.
        float: The area of the second polygon.
        float: The fractional difference in vertices between the second and
                first polygon.
        float: The fraction difference in area between the second and first
                polygon.
    """

    # Compute the areas of the two polygons
    area1 = _compute_area(points1)
    area2 = _compute_area(points2)

    # Compute the fractional difference in vertices and area
    (num_vertices1, _) = points1.shape
    (num_vertices2, _) = points2.shape
    vertex_diff = (num_vertices2 - num_vertices1) / float(num_vertices1)
    area_diff = (area2 - area1) / area1

    return (area1, area2, vertex_diff, area_diff)

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

def _polyline_approx(points, thickness_threshold):
    """Approximates the given points using a thick-polyline approximation.

    Given a polyline defined by the given sorted points, and a thickness that
    defines a dominant curve, this function finds the approximation of the
    polyline containing only dominant points.
    """

    # If there is only one or two points, then we have our dominant point(s)
    assert(points.ndim == 2 and points.shape[1] == 2)
    (num_points, _) = points.shape
    assert(num_points > 1)
    if num_points <= 2:
        return _points_to_ordered_dict(points)

    # Compute the line between the two endpoints of the polyline, then normalize
    # the standard form so it can be used to directly compute distances.
    ((x0, y0), (x1, y1)) = (points[0, :], points[-1, :])
    (a_value, b_value) = (y1 - y0, -(x1 - x0))
    c_value = -(a_value * x0 + b_value * y0)
    ab_norm = numpy.linalg.norm([a_value, b_value])
    line_vector = numpy.array([a_value, b_value, c_value]) / ab_norm

    # Find the extreme points in the set of points above and below the line. The
    # points are partitioned based on their signed distance from the line.
    point_distances = numpy.dot(points, line_vector[0:2]) + line_vector[2]
    below_mask = point_distances < 0
    below_extremum_index = _find_extremum(points, point_distances, below_mask)
    above_extremum_index = _find_extremum(points, point_distances, ~below_mask)

    # Compute the thickness based on the distance of the extreme points from the
    # line. If it is above the threshold, the endpoints are dominant.
    thickness = abs(_get_array_value(point_distances, below_extremum_index,
                        0.0))
    thickness += abs(_get_array_value(point_distances, above_extremum_index,
                        0.0))

    if thickness < thickness_threshold:
        return _points_to_ordered_dict(points[(0, -1), :])

    # Otherwise, partition the points based on the extreme values. Recursively
    # approximate the smaller polylines represented by these points.
    (points1, points2, points3) = _partition_points(points,
            below_extremum_index, above_extremum_index)
    dominant_points = _polyline_approx(points1, thickness_threshold)
    dominant_points.update(_polyline_approx(points2, thickness_threshold))
    if points3 is not None:
        dominant_points.update(_polyline_approx(points3, thickness_threshold))

    return dominant_points

def _points_to_ordered_dict(points):
    """Converts the given points to an ordered dictionary, treating the columns
    as points, which are used as keys."""

    (num_points, _) = points.shape
    generator = ((tuple(points[i, :]), None) for i in range(num_points))
    return OrderedDict(generator)

def _find_extremum(points, point_distances, subset_mask):
    """Finds the index of the minimum or maximum point from the given subset of
    the points. The endpoints of point list are excluded."""

    # Exclude the first and last points (endpoints) from the subset.
    points_excluded = points[1:-1, :]
    point_distances_excluded = point_distances[1:-1]
    subset_mask_excluded = subset_mask[1:-1]

    # If the point set is empty, then no extremum exists.
    points_subset = points_excluded[subset_mask_excluded, :]
    point_distances_subset = point_distances_excluded[subset_mask_excluded]
    if points_subset.size == 0:
        return None

    # Otherwise, find the extreme value based on the absolute distance from the
    # line. Make sure to account for the missing first point in the index.
    subset_indices = numpy.where(subset_mask_excluded)
    extremum_index = numpy.argmax(numpy.absolute(point_distances_subset))
    return subset_indices[0][extremum_index] + 1

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

    (num_points, _) = points.shape
    assert(below_extremum_index is not None or above_extremum_index is not None)
    assert(above_extremum_index != below_extremum_index)
    assert(below_extremum_index is None or (below_extremum_index > 0 and
            below_extremum_index < num_points - 1))
    assert(above_extremum_index is None or (above_extremum_index > 0 and
            above_extremum_index < num_points - 1))

    # Partition the points based on which extrema doesn't exist. If both do,
    # then partition, making sure to respect the ordering.
    if below_extremum_index is None:
        return (
            points[0:above_extremum_index+1, :],
            points[above_extremum_index:, :],
            None,
        )
    elif above_extremum_index is None:
        return (
            points[0:below_extremum_index+1, :],
            points[below_extremum_index:, :],
            None,
        )
    elif below_extremum_index < above_extremum_index:
        return (
            points[0:below_extremum_index+1, :],
            points[below_extremum_index:above_extremum_index+1, :],
            points[above_extremum_index:, :],
        )
    else:
        return (
            points[0:above_extremum_index+1, :],
            points[above_extremum_index:below_extremum_index+1, :],
            points[below_extremum_index:, :],
        )

def _compute_area(points):
    """Computes the area of the polygon given by the specified points."""

    # Put each pair of vertices into a matrix, and stack them
    points_shifted = numpy.roll(points, -1, axis=0)
    vertex_pairs = numpy.stack([points, points_shifted], axis=2)

    # The signed area of the polygon is half of the absolute value of the sum of
    # the determinants of the vertex pairs.
    vertex_areas = numpy.linalg.det(vertex_pairs)
    return 1 / 2.0 * abs(numpy.sum(vertex_areas))
