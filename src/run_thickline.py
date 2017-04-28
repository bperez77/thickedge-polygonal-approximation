#! /usr/bin/env python2

import numpy as np
import sys
from polygonal_approximation import thick_polygonal_approximate
from background_subtract import process_video, process_camera


def main():
    if len(sys.argv) < 3:
        print 'usage: %s <video_file> <thickness>' % sys.argv[0]
        exit(1)

    video_file = sys.argv[1]
    thickness = sys.argv[2]

    thick_polygon = []
    frames = process_video(video_file)

    for frame in frames:
        for poly in frame:
            # Get array into the right shape
            poly = np.reshape(poly, (poly.shape[0], poly.shape[2])).T
            thick_polygon.append(thick_polygonal_approximate(poly, thickness))

    # Do something with the thick polygon

if __name__ == "__main__":
    main()
