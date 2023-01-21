import numpy as np
import cv2
import time
import os
from pose_estimation_tf import PoseEstimator
from calorie_estimation_tf import CalorieEstimator
import argparse
import pygame
from threading import Thread

class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

class PoseGet:
    def __init__(self, rotate_code):
        self.pose = PoseEstimator()
        self.calorie = CalorieEstimator()
        self.points_prev = None
        self.t = time.time()
        self.total_calories = 0
        self.rotate_code = rotate_code
        self.overlay_frame = None
        self.stopped = False

    def start(self, video_obj):
        self.overlay_frame = video_obj.frame
        Thread(target=self.get, args=(video_obj,)).start()
        return self

    def get(self, video_obj):
        while not self.stopped:
            ret, frame = video_obj.grabbed, video_obj.frame
            dt = time.time() - self.t
            self.t = time.time()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return

            if self.rotate_code is not None:
                frame = cv2.rotate(frame, self.rotate_code) 

            points, valid = self.pose.estimate(frame)

            # Draw Skeleton
            for pair in self.pose.pose_pairs:
                p1, p2 = pair

                if valid[p1] and valid[p2]:
                    cv2.line(frame, points[p1,:], points[p2,:], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                    cv2.circle(frame, points[p1,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[p2,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
            if self.points_prev is not None and valid.all():
                self.total_calories += self.calorie.estimate(self.points_prev, points, dt)
                cv2.putText(frame, "Calories Burnt = {:.2f}".format(self.total_calories), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
                print(self.total_calories)
            
            self.overlay_frame = frame
            self.points_prev = points
    
    def stop(self):
        self.stopped = True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('file', help='path to the dataset folder containing rgb/ and depth/', nargs='*')

    args = parser.parse_args()

    if len(args.file) == 0:
        # Webcam Capture
        rotate_code = None
        video_getter = VideoGet().start()
        pose_getter = PoseGet(rotate_code).start(video_getter)
        # output_path = '../output/webcam.mp4'
    else:
        # Video File Capture
        rotate_code = cv2.ROTATE_180
        video_getter = VideoGet(args.file[0]).start()
        pose_getter = PoseGet(rotate_code).start(video_getter)
        # output_path = '../output/' + os.path.splitext(os.path.split(args.file[0])[1])[0] + '_pose.mp4'
    
    if not video_getter.stream.isOpened():
        raise Exception("VideoCapture object cannot be opened")

    # frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # assert fps > 10
    
    # vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    pygame.init()
    size = (1024, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("User Live Feed")

    WHITE = (255, 255, 255)
    GREY = (127, 127, 127)
    RED = (255, 0, 0)
    DARK_RED = (139, 0, 0)
    GREEN = (0,255,0)
    BLACK = (0, 0, 0)
    STOP_COLOR = RED

    screen.fill(WHITE)
    pygame.display.flip()

    windowOpen = True
    font = pygame.font.SysFont('Calibri', 30, True, False)

    while windowOpen:
        mouse = pygame.mouse.get_pos()

        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                windowOpen = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if 0 <= mouse[0] <= 100 and 400 <= mouse[1] <= 600:
                    STOP_COLOR = DARK_RED
            elif event.type == pygame.MOUSEBUTTONUP:
                if 0 <= mouse[0] <= 100 and 400 <= mouse[1] <= 600:
                    pygame.quit()
        
        frame = video_getter.frame
        # frame = pose_getter.overlay_frame
        frame = cv2.resize(frame, (1024, 600))

        video_surf = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")

        pygame.draw.rect(video_surf, GREY, [0, 0, 100, 200])
        text = font.render(f"Calories: {pose_getter.total_calories}", True, BLACK)
        text = pygame.transform.rotate(text, 270)
        video_surf.blit(text, [45, 20])

        pygame.draw.rect(video_surf, GREEN, [0, 200, 100, 200])
        text = font.render(f"Discount: TODO", True, BLACK)
        text = pygame.transform.rotate(text, 270)
        video_surf.blit(text, [45, 220])

        pygame.draw.rect(video_surf, STOP_COLOR, [0, 400, 100, 200])
        text = font.render("Pay", True, BLACK)
        text = pygame.transform.rotate(text, 270)
        video_surf.blit(text, [45, 450])

        screen.blit(video_surf, (0, 0))

        pygame.display.flip()

    # vid_writer.release()

if __name__ == '__main__':
    main()