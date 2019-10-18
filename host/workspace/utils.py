import logging
import argparse
import cv2


def get_logger():
    # init:
    logger = logging.getLogger('TfPoseEstimator-WebCam')
    # config:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_parser():
    """ Build arg parser for intelli-train frontend
    """
    # init:
    parser = argparse.ArgumentParser(
        description='intelli-train-motion-evaluation realtime camera'
    )

    # camera device:
    parser.add_argument(
        '--camera', 
        type=int, default=0,
        help='set camera ID for real-time motion evaluation.'
    )
    # video input:
    parser.add_argument(
        '--video', 
        type=str, default='set-00.mp4',
        help='select motion video for reference generation. must be file in reference/video'
    )
    # video input:
    parser.add_argument(
        '--reference', 
        type=str, default='set-00.json',
        help='reference motion for real-time evaluation. must be file in reference/json'
    )

    # time to prepare:
    parser.add_argument(
        '--prep-time', 
        type=float, default=20.0,
        help='time to prepare before evaluation starts.'
    )
    # time to prepare:
    parser.add_argument(
        '--d-max', 
        type=float, default=1.57,
        help='max allowable distance.'
    )
    parser.add_argument(
        '--s-min', 
        type=float, default=20,
        help='base score.'
    )

    # model input size:
    parser.add_argument(
        '--resize', 
        type=str, default='320x240',
        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736'
    )
    parser.add_argument(
        '--resize-out-ratio', type=float, default=4.0,
        help='if provided, resize heatmaps before they are post-processed. default=1.0'
    )
    parser.add_argument(
        '--input-mode', type=str, default='left', choices=['left', 'right', 'stereo'], 
        help='set input mode for stereo camera. left / right / stereo. default=left'
    )
    # model:
    parser.add_argument(
        '--model', 
        type=str, default='mobilenet_thin', choices=['cmu', 'mobilenet_thin', 'mobilenet_v2_large', 'mobilenet_v2_small'],
        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small'
    )
    # tensor RT:
    parser.add_argument(
        '--tensorrt', 
        type=bool, default=False, 
        help='for tensorrt process.'
    )
    # debug:
    parser.add_argument(
        '--show-process', 
        type=bool, default=False,
        help='for debug purpose, if enabled, inference speed would drop.'
    )

    return parser


def shall_terminate():
    """ Detect termination signal
    """
    return cv2.waitKey(1) == 27