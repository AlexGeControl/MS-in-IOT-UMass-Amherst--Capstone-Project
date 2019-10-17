import sys
import cv2
import os
from sys import platform
from openpose import pyopenpose as op

def set_params():
    """ config openpose
    """
    params = dict()
    params["model_folder"] = "/openpose/models/"
    return params

def main():
    params = set_params()

    # init OpenPose instance:
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # open camera:
    stream = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret,img = stream.read()

        # Output keypoints and the image with the human skeleton blended on it
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])

        # Display the stream
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()