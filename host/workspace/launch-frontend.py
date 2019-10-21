import os
import time
import datetime

import cv2
import json
import numpy as np
import similaritymeasures
from scipy.interpolate import CubicSpline

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from utils import *


logger = get_logger()


class Camera(object):
    __slots__ = (
        'device_id', 
        'W', 
        'H', 
        'input_mode', 
        'last_access_time',
        'FPS',
        'show_process', 
        'camera'
    )

    def __init__(self, args):
        """ Init camera handler:
        """
        # init handler:
        self.device_id = args.camera
        self.camera = cv2.VideoCapture(self.device_id)
        # set input mode:
        self.input_mode = args.input_mode
        # set input resolution:
        self.W = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # init access time:
        self.last_access_time = time.time()
        self.FPS = 0
        # debugger:
        self.show_process = args.show_process
        if (self.show_process):
            logger.debug(
                '[Camera]: Open device %d' % self.device_id
            )
            logger.info(
                '[Camera]: Input resolution: %dx%d' % (self.W, self.H)
            )
        
    def read(self):
        """ Read frame from camera
        """
        try:
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
        except Exception as e:
            print(e)
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
        cv2.imshow('Real-Time Pose Estimation', image)

    def tear_down(self):
        """ Close all windows
        """
        cv2.destroyAllWindows()
        if (self.show_process):
            logger.debug('[Canvas]: Finished.')

class MotionEvaluator(object):
    __slots__ = (
        'reference',
        'prep_time',
        'start_time',
        'frame_index',
        'timestamp',
        'trajectories',
        'd_max',
        's_min',
        'current_distance',
        'current_score',
        'scores',
        'show_process'
    )

    def __init__(self, args):
        # load motion reference:
        input_filename = os.path.join("reference/json", args.reference)
        with open(input_filename, 'r') as input_reference: 
            self.reference = json.load(input_reference)
        self.prep_time = args.prep_time
        self.d_max = args.d_max
        self.s_min = args.s_min
        self.current_distance = self.d_max
        self.current_score = self.s_min
        # init time reference:
        self.start_time = 0
        self.frame_index = 0
        self.timestamp = 0.0
        # set trajectory container:
        self.trajectories = {
            i: {"t": [], "x": [], "y": []} for i in range(
                common.CocoPart.RShoulder.value,
                common.CocoPart.LAnkle.value + 1
            )
        }
        self.scores = []
        self.show_process = args.show_process
    
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

    def _evaluate(self):
        # init:
        distances = []

        # calculate distances:
        FPS = int(self.reference["FPS"])
        for i in self.trajectories.keys():
            if (len(self.trajectories[i]["t"]) < FPS):
                continue

            # init evaluation interval:
            ref_xs = np.asarray(self.trajectories[i]["t"][-FPS:])
            eval_xs_start_index = int(FPS * ref_xs[+0]) + 1
            eval_xs_end_index = int(FPS * ref_xs[-1]) + 1
            eval_xs = (1.0 / FPS) * np.arange(
                eval_xs_start_index,
                eval_xs_end_index
            )

            # init distance:
            distance = 0.0
            for k in ["x", "y"]:
                # for each axis, smooth observation with cubic spline:
                ref_ys = np.asarray(self.trajectories[i][k][-FPS:])
                ref_cs = CubicSpline(ref_xs, ref_ys)
                eval_ys = ref_cs(eval_xs)
                # save table for motion reference:
                distance += similaritymeasures.frechet_dist(
                    self.reference["reference"][str(i)][k][eval_xs_start_index: eval_xs_end_index],
                    eval_ys
                )

            distances.append(distance)
        
        # return weighed average:
        return np.average(distances, weights=distances)

    def _score(self, distance):
        if distance > self.d_max:
            return 0

        return (self.d_max - distance) / self.d_max * (100.0 - self.s_min) + self.s_min

    def evaluate(self, humans):
        """ Add new observation
        """
        # update timestamp:
        if self.start_time == 0: 
            self.start_time = time.time()
            self.timestamp = 0.0
        else:
            self.timestamp = time.time() - self.start_time

        # if timeout, return None
        if (self.timestamp >= self.prep_time + self.reference["duration"]):
            return None
        elif (self.timestamp < self.prep_time):
            return "Get READY within %d Seconds!" % (
                int(self.prep_time - self.timestamp)
            )

        if len(humans) == 0:
            return

        # TODO: enable multiple exerciser
        human = humans[0]

        # extract joint positions:
        for (i, (x, y)) in self._scale(human):
            self.trajectories[i]["t"].append(self.timestamp - self.prep_time)
            self.trajectories[i]["x"].append(x)
            self.trajectories[i]["y"].append(y)

        # evaluate at given interval:
        self.frame_index += 1
        if (self.frame_index == self.reference["FPS"]):
            self.frame_index = 0
            # get average distance:
            self.current_distance = self._evaluate()
            self.current_score = self._score(self.current_distance)

            self.scores.append(
                {
                    "timestamp": self.timestamp - self.prep_time,
                    "score": self.current_score
                }
            )

        return "Current Score: %d" % (
            int(self.current_score)
        )

    def save(self):
        """ Save evaluation as JSON
        """        
        filename = "%s-%s.json" % (
            self.reference["name"],
            datetime.datetime.now().strftime("%B-%d-%Y--%I:%M%p")
        )
        # add dir name:
        filename = os.path.join("evaluation", filename)
        with open(filename, 'w') as output_evaluation:
            json.dump(self.scores, output_evaluation)

if __name__ == '__main__':
    # parse arguments:
    args = get_parser().parse_args()

    # init camera:
    camera = Camera(args)
    # init model:
    estimator = PoseEstimator(args)
    # init canvas:
    canvas = Canvas(args)
    # init evaluator:
    evaluator = MotionEvaluator(args)

    while True:
        # fetch frame:
        image = camera.read()
        if image is None:
            break

        # inference:
        humans, image_with_pose_drawn = estimator.inference(
            image = image, 
            draw_pose=True
        )
    
        # get captain:
        captain = evaluator.evaluate(humans)
        if captain is None:
            break

        # show prompt:
        canvas.show(
            image = image_with_pose_drawn,
            captain = captain
        )

        if shall_terminate():
            break

    # save evaluation:
    evaluator.save()
    canvas.tear_down()
