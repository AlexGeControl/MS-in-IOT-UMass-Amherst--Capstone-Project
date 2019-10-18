import os
import time

import cv2
import json
import numpy as np
from scipy.interpolate import CubicSpline

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from utils import *


logger = get_logger()


class Video(object):
    __slots__ = (
        'name', 
        'W', 
        'H', 
        'FPS',
        'show_process', 
        'video'
    )

    def __init__(self, args):
        """ Init camera handler:
        """
        # init handler:
        self.name = os.path.splitext(args.video)[0]
        filename = os.path.join("reference/video", args.video)
        self.video = cv2.VideoCapture(filename)
        # get video properties:
        self.W = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = self.video.get(cv2.CAP_PROP_FPS)        
        self.show_process = args.show_process
        # debugger:
        if (self.show_process):
            logger.debug(
                '[Video]: Open video %s' % args.video
            )
            logger.info(
                '[Video]: Input resolution: %dx%d, FPS: %f' % (self.W, self.H, self.FPS)
            )

    def read(self):
        """ Read frame from camera
        """
        try:
            ret, image = self.video.read()            
            return image
        except Exception as e:
            return None


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
        self.use_tensorrt = args.tensorrt
        self.model = args.model
        self.input_size = model_wh(args.resize)
        self.upsample_rate = args.resize_out_ratio
        self.estimator = TfPoseEstimator(
            get_graph_path(self.model), 
            target_size=self.input_size, 
            trt_bool=self.use_tensorrt
        )
        self.show_process = args.show_process
        if (self.show_process):
            logger.debug(
                '[PoseEstimator]: Initialization %s : %s' % (args.model, get_graph_path(args.model))
            )

    def inference(self, image, draw_pose=False):
        """ Inference pose from input image 
        """
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

        if (self.show_process):
            logger.debug(
                '[PoseEstimator]: Pose estimation+'
            )

        return humans, image


class Canvas(object):
    __slots__ = ( 
        'show_process'
    )

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
        cv2.imshow('Motion Reference Generation', image)

    def tear_down(self):
        """ Close all windows
        """
        cv2.destroyAllWindows()
        if (self.show_process):
            logger.debug('[Canvas]: Finished.')

class ReferenceGenerator(object):
    __slots__ = (
        'frame_index',
        'trajectories'
    )

    def __init__(self, args):
        self.frame_index = 0
        self.trajectories = {
            i: {"i": [], "x": [], "y": []} for i in range(
                common.CocoPart.RShoulder.value,
                common.CocoPart.LAnkle.value + 1
            )
        }
    
    def _scale(self, human):
        frame = {
            "i": [],
            "x": [],
            "y": []
        }
        for i in range(
            common.CocoPart.RShoulder.value,
            common.CocoPart.LAnkle.value + 1
        ):
            # if not detected:
            if i not in human.body_parts.keys():
                continue
            # if detected:
            body_part = human.body_parts[i]
            x, y = (body_part.x, body_part.y)
            # add:
            frame["i"].append(i)
            frame["x"].append(x)
            frame["y"].append(y)

        # scale:
        for k in ["x", "y"]:
            frame[k] = np.asarray(frame[k])
            frame[k] = np.interp(
                frame[k], 
                (frame[k].min(), frame[k].max()), 
                (-1, +1)
            )
        
        return zip(
            frame["i"], 
            zip(frame["x"], frame["y"])
        )

    def append(self, humans):
        """ Add new observation
        """
        if len(humans) == 0:
            return

        # TODO: enable multiple exerciser
        human = humans[0]

        # extract joint positions:
        for (i, (x, y)) in self._scale(human):
            self.trajectories[i]["i"].append(self.frame_index)
            self.trajectories[i]["x"].append(x)
            self.trajectories[i]["y"].append(y)

        self.frame_index += 1

    def save(self, ref_video, eval_FPS):
        """ Save trajectories as JSON
        """        
        ref_timestep = 1.0 / ref_video.FPS
        eval_timestep = 1.0 / eval_FPS

        # set evaluation time intervals:
        eval_ts = np.arange(
            start = 0.0, stop = ref_timestep * self.frame_index, 
            step = eval_timestep
        )

        # init motion reference output:
        motion_reference = {}
        motion_reference["duration"] = eval_ts[-1]
        motion_reference["FPS"] = eval_FPS
        motion_reference["reference"] = {}

        for i in self.trajectories.keys():
            motion_reference["reference"][i] = {}

            ref_xs = ref_timestep * np.asarray(self.trajectories[i]["i"])
            for k in ["x", "y"]:
                # for each axis, smooth observation with cubic spline:
                ref_ys = np.asarray(self.trajectories[i][k])
                ref_cs = CubicSpline(ref_xs, ref_ys)
                eval_ys = ref_cs(eval_ts)
                # save table for motion reference:
                motion_reference["reference"][i][k] = eval_ys.tolist()

        # save:
        output_filename = os.path.join("reference/json", "%s.json" % (video.name))
        with open(output_filename, 'w') as output_motion_reference:
            json.dump(motion_reference, output_motion_reference)

if __name__ == '__main__':
    # parse arguments:
    args = get_parser().parse_args()

    # init video:
    video = Video(args)
    # init model:
    estimator = PoseEstimator(args)
    # init canvas:
    canvas = Canvas(args)
    # init evaluator:
    generator = ReferenceGenerator(args)

    while True:
        # fetch frame:
        image = video.read()
        if image is None:
            break

        # inference:
        humans, image_with_pose_drawn = estimator.inference(
            image = image, 
            draw_pose=True
        )
        # add new observation:
        generator.append(humans)

        # show prompt:
        canvas.show(
            image = image_with_pose_drawn,
            captain = "Motion Ref. Gen."
        )

        if shall_terminate():
            break

    # generate motion reference:
    generator.save(
        ref_video = video,
        eval_FPS = 10.0
    )
    canvas.tear_down()
