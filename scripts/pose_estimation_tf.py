# Credit for some implementation of tflite model: https://github.com/ecd1012/rpi_pose_estimation/blob/main/

import numpy as np
import cv2
import time
import os
import argparse
# import tflite_runtime as tf
# from tflite_runtime.interpreter import Interpreter
from tensorflow.lite.python.interpreter import Interpreter

class PoseEstimator:
    def __init__(self, model='lightning'):
        # Specify the paths for the 2 files
        model_path = '../models/movenet/model.tflite'
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.output_stride = 4
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.input_size = 256
        assert self.height == self.input_size and self.width == self.input_size

        self.keypoint_dict = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

        self.pose_pairs = [
            [0,5],
            [0,6],
            [5,7],
            [6,8],
            [7,9],
            [8,10],
            [5,11],
            [6,12],
            [11,12],
            [11,13],
            [12,14],
            [13,15],
            [14,16]
        ]

    def _keypoints_for_display(self, keypoints_with_scores, height, width, keypoint_threshold=0.11):
        """Returns high confidence keypoints and edges for visualization.

        Args:
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
            height: height of the image in pixels.
            width: width of the image in pixels.
            keypoint_threshold: minimum confidence score for a keypoint to be
            visualized.

        Returns:
            A (keypoints_xy, edges_xy, edge_colors) containing:
            * the coordinates of all keypoints of all detected entities;
            * the coordinates of all skeleton edges of all detected entities;
            * the colors in which the edges should be plotted.
        """
        kpts_x = keypoints_with_scores[..., :, 1]
        kpts_y = keypoints_with_scores[..., :, 0]
        kpts_scores = keypoints_with_scores[..., :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1).astype(int)
        valid_kpts = kpts_scores > keypoint_threshold
        return kpts_absolute_xy.squeeze(), valid_kpts.squeeze()

    def estimate(self, frame, threshold=0.11):
        frame_height, frame_width = frame.shape[:2]

        frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)
        input_data = input_data.astype(np.uint8)
        
        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])

        return self._keypoints_for_display(keypoints_with_scores, frame_height, frame_width, threshold)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('vid_path', help='Video File Path for Pose Tracking', nargs='*')

    args = parser.parse_args()

    if len(args.vid_path) == 0:
        # Webcam Capture
        cap = cv2.VideoCapture(0)
        rotate_code = None
        output_path = '../output/webcam.mp4'
    else:
        # Video File Capture
        cap = cv2.VideoCapture(args.vid_path[0])
        rotate_code = cv2.ROTATE_180
        output_path = '../output/' + os.path.splitext(os.path.split(args.vid_path[0])[1])[0] + '_pose.mp4'
    
    if not cap.isOpened():
        raise Exception("VideoCapture object cannot be opened")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 10
    
    vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    pose = PoseEstimator()
    
    while True:
        t = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code) 

        points, valid = pose.estimate(frame)

        # Draw Skeleton
        for pair in pose.pose_pairs:
            p1, p2 = pair

            if valid[p1] and valid[p2]:
                cv2.line(frame, points[p1,:], points[p2,:], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[p1,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[p2,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        
        # for i, point in enumerate(points):
        #     # cv2.circle(frame, point, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #     cv2.putText(frame, str(i), point, cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        
        vid_writer.write(frame)

        cv2.imshow('Output Pose', frame)
        if cv2.waitKey(1) != -1:
            break

    vid_writer.release()

if __name__ == '__main__':
    main()
