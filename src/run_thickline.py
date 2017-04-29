#! /usr/bin/env python2

import numpy as np
import sys
import cv2
from polygonal_approximation import thick_polygonal_approximate
from background_subtract import process_video, process_camera

def main():
    if len(sys.argv) < 3:
        print 'usage: %s <video_file> <thickness>' % sys.argv[0]
        exit(1)

    video_file = sys.argv[1]
    thickness = int(sys.argv[2])

    thick_polygons = []
    frames = process_video(video_file)

    i = 0
    for frame in frames:
        polygons = []
        for poly in frame:
            # Get array into the right shape
            poly = np.reshape(poly, (poly.shape[0], poly.shape[2]))
            polygons.append(thick_polygonal_approximate(poly, thickness))
        thick_polygons.append(polygons)

    # Do something with the thick polygon
    capture = cv2.VideoCapture(video_file)
    if not capture.isOpened():
        print("Could not open file: ", video_file)

    pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    i = 0
    while True:
        flag, frame = capture.read()
        pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

        if len(thick_polygons[i]) > 0:
            cv2.polylines(frame, np.int32([thick_polygons[i][0]]), True, (10,150,220), thickness=thickness)

        cv2.imshow(video_file, frame)

        i += 1

        if cv2.waitKey(50) > 0:
            break

        # End of video
        if pos_frame == capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
