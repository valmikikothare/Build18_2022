import numpy as np
import cv2
import time
import os
from pose_estimation import PoseEstimator
import argparse
import tensorflow as tf

class CalorieEstimator:
    def __init__(self):
        self.d_scale = 0.002
        self.m_scale = 75
        head_mass = 3
        neck_mass = 1
        shoulder_mass = 3
        elbow_mass = 2
        hand_mass = 1
        hip_mass = 10
        knee_mass = 5
        foot_mass = 2
        torso_mass = 10
        self.mass = np.array([head_mass,
                            neck_mass,
                            shoulder_mass,
                            elbow_mass,
                            hand_mass,
                            shoulder_mass,
                            elbow_mass,
                            hand_mass,
                            hip_mass,
                            knee_mass,
                            foot_mass,
                            hip_mass,
                            knee_mass,
                            foot_mass,
                            torso_mass])
        self.mass = self.m_scale*(self.mass/self.mass.mean())


    def estimate(self, points_prev, points_cur, dt):
        v2 = (self.d_scale*(np.linalg.norm(points_cur - points_prev, axis=1))/dt)**2
        e = self.mass * v2
        return e.sum()/4184

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('file', help='path to the dataset folder containing rgb/ and depth/', nargs='*')

    args = parser.parse_args()

    if len(args.file) == 0:
        # Webcam Capture
        cap = cv2.VideoCapture(0)
        rotate_code = None
        # output_path = '../output/webcam.mp4'
    else:
        # Video File Capture
        cap = cv2.VideoCapture(args.file[0])
        rotate_code = cv2.ROTATE_180
        # output_path = '../output/' + os.path.splitext(os.path.split(args.file[0])[1])[0] + '_pose.mp4'
    
    if not cap.isOpened():
        raise Exception("VideoCapture object cannot be opened")

    # frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # assert fps > 10
    
    # vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    pose = PoseEstimator(mode='mpi')
    calorie = CalorieEstimator()

    points_prev = None
    t = time.time()
    total_calories = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow('Output Pose', frame)
        if cv2.waitKey(1) != -1:
            break
        dt = time.time() - t
        t = time.time()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code) 

        points = pose.estimate(frame)

        # Draw Skeleton
        # for pair in pose.pose_pairs:
        #     partA = pair[0]
        #     partB = pair[1]

        #     if (points[partA,:] >= 0).all() and (points[partB,:] >= 0).all():
        #         cv2.line(frame, points[partA,:], points[partB,:], (0, 255, 255), 3, lineType=cv2.LINE_AA)
        #         cv2.circle(frame, points[partA,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #         cv2.circle(frame, points[partB,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        
        if points_prev is not None:
            total_calories += calorie.estimate(points_prev, points, dt)
            cv2.putText(frame, "Calories Burnt = {:.2f}".format(total_calories), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
            print(total_calories)
        
        points_prev = points
        
        # vid_writer.write(frame)

        cv2.imshow('Output Pose', frame)
        if cv2.waitKey(1) != -1:
            break

    # vid_writer.release()

if __name__ == '__main__':
    main()
