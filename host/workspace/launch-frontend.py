import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from utils import *


logger = get_logger()


class Camera(object):
    __slots__ = (
        'device_id', 
        'W', 
        'H', 
        'last_access_time',
        'FPS',
        'input_mode', 
        'show_process', 
        'camera'
    )

    def __init__(self, args):
        """ Init camera handler:
        """
        self.show_process = args.show_process
        if (self.show_process):
            logger.debug(
                '[Camera]: Open device %d' % args.camera
            )
        # init handler:
        self.device_id = args.camera
        self.camera = cv2.VideoCapture(self.device_id)
        # set input resolution:
        ret, image = self.camera.read()
        self.W, self.H = image.shape[1], image.shape[0]
        # init access time:
        self.last_access_time = time.time()
        self.FPS = 0
        if (self.show_process):
            logger.info(
                '[Camera]: Input resolution: %dx%d' % (self.W, self.H)
            )
        # set input mode:
        self.input_mode = args.input_mode
    
    def read(self):
        """ Read frame from camera
        """
        ret, image = self.camera.read()

        # update FPS:
        self.FPS = 1.0 / (time.time() - self.last_access_time)
        self.last_access_time = time.time()

        # crop stereo input:
        if self.input_mode == "left":
            return image[0:self.H, int(self.W/2):self.W]
        elif self.input_mode == "right":
            return image[0:self.H, 0:int(self.W/2)]
        
        return image


class PoseEstimator(object):
    __slots__ = (
        'use_tensorrt', 
        'model', 
        'input_size', 
        'upsample_rate', 
        'show_process', 
        'estimator'
    )

    def __init__(self, args): 
        """ Init pose estimator
        """
        self.show_process = args.show_process
        if (self.show_process):
            logger.debug(
                '[PoseEstimator]: Initialization %s : %s' % (args.model, get_graph_path(args.model))
            )

        self.use_tensorrt = args.tensorrt
        self.model = args.model
        self.input_size = model_wh(args.resize)
        self.upsample_rate = args.resize_out_ratio

        self.estimator = TfPoseEstimator(
            get_graph_path(self.model), 
            target_size=self.input_size, 
            trt_bool=self.use_tensorrt
        )

    def inference(self, image, draw_pose=False):
        """ Inference pose from input image 
        """
        if (self.show_process):
            logger.debug(
                '[PoseEstimator]: Pose estimation+'
            )
        humans = self.estimator.inference(
            image, 
            resize_to_default=True, 
            upsample_size=self.upsample_rate
        )

        if (draw_pose):
            if (self.show_process):
                logger.debug(
                    '[PoseEstimator]: Post-process+'
                )
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        return humans, image


class Canvas(object):
    def __init__(self, args): 
        """ Init pose estimator
        """
        self.show_process = args.show_process
    
    def show(self, image ,captain):
        """ Draw image with captain
        """
        if (self.show_process):
            logger.debug('[Canvas]: Show+')

        cv2.putText(
            image,
            captain,
            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2
        )
        cv2.imshow('Real-Time Pose Estimation', image)

    def tear_down(self):
        """ Close all windows
        """
        cv2.destroyAllWindows()
        if (self.show_process):
            logger.debug('[Canvas]: Finished.')


if __name__ == '__main__':
    # parse arguments:
    args = get_parser().parse_args()

    # init model:
    estimator = PoseEstimator(args)
    # init camera:
    camera = Camera(args)
    # init canvas:
    canvas = Canvas(args)

    while True:
        # fetch frame:
        image = camera.read()
        # inference:
        humans, image_with_pose_drawn = estimator.inference(
            image = image, 
            draw_pose=True
        )
        # show prompt:
        canvas.show(
            image = image_with_pose_drawn,
            captain = "FPS: %f" % camera.FPS
        )

        if shall_terminate():
            break

    canvas.tear_down()
