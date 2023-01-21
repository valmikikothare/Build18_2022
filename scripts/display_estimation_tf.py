import numpy as np
import cv2
import time
import os
from pose_estimation_tf import PoseEstimator
from calorie_estimation_tf import CalorieEstimator
import argparse
import pygame
from threading import Thread

WHITE = (255, 255, 255)
LIGHT_GREY = (220, 220, 220)
GREY = (127, 127, 127)
RED = (255, 0, 0)
DARK_RED = (139, 0, 0)
GREEN = (0,255,0)
BLACK = (0, 0, 0)
STOP_COLOR = RED
BUTTON_WINDOW = 0
VIDEO_WINDOW = 1

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

    def read(self):
        return self.frame

class PoseGet:
    def __init__(self, rotate_code=cv2.ROTATE_90_CLOCKWISE, render_frame=False):
        self.pose = PoseEstimator()
        self.calorie = CalorieEstimator()
        self.points_prev = None
        self.t = time.time()
        self.total_calories = 0
        self.rotate_code = rotate_code
        self.overlay_frame = None
        self.stopped = False
        self.window_size = 5
        self.window_x = []
        self.window_y = []
        self.window_t = []
        self.render_frame = render_frame

    def start(self, video_obj):
        self.overlay_frame = video_obj.read()
        Thread(target=self.get, args=(video_obj,)).start()
        return self

    def get(self, video_obj):
        while not self.stopped:
            ret, frame = video_obj.grabbed, video_obj.frame
            self.window_t.append(time.time())

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return

            if self.rotate_code is not None:
                frame = cv2.rotate(frame, self.rotate_code) 

            points, valid = self.pose.estimate(frame)

            # Draw Skeleton
            if self.render_frame:
                for pair in self.pose.pose_pairs:
                    p1, p2 = pair
                    if valid[p1] and valid[p2]:
                        cv2.line(frame, points[p1,:], points[p2,:], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                        cv2.circle(frame, points[p1,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                        cv2.circle(frame, points[p2,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
            if valid.all():
                self.window_x.append(points[:,0])
                self.window_y.append(points[:,1])
                if len(self.window_x) == self.window_size:
                    window_x = np.column_stack(self.window_x)
                    window_y = np.column_stack(self.window_y)
                    dx = (window_x[:,1:] - window_x[:,:-1]).mean(axis=1)
                    dy = (window_y[:,1:] - window_y[:,:-1]).mean(axis=1)
                    displacement = np.column_stack(dx, dy)
                    dt = (self.window_t[-1] - self.window_t[0]) / self.window_size
                    self.total_calories += self.calorie.estimate(self.points_prev, displacement, dt)
                    self.window_x.pop(0)
                    self.window_y.pop(0)
                    self.window_t.pop(0)
            
            self.overlay_frame = frame
            self.points_prev = points
    
    def stop(self):
        self.stopped = True

    def read_calories(self):
        return self.total_calories
        
    def read_frame(self):
        return self.overlay_frame

    def read_frame_and_calories(self):
        return self.overlay_frame, self.total_calories

class ButtonWindow():
    # Create the buttons
    width, height = (1024, 600)
    b_w = width/4.5
    b_h = height/10
    spacing = b_w/3
    button_1 = pygame.Rect(width/2 - b_w/2 - spacing - b_w, height/2-b_h/2, b_w, b_h)
    button_2 = pygame.Rect(width/2 - b_w/2, height/2-b_h/2, b_w, b_h)
    cancel_button = pygame.Rect(width/2 + b_w/2 + spacing, height/2-b_h/2, b_w, b_h)
    confirm_button = pygame.Rect(width/2 - b_w/2, height/2+b_h/2+spacing, b_w, b_h)

    # Selection Menu Title
    title = title_font.render("Please select your desired snack :)", True, BLACK)
    title_rect = title.get_rect()
    title_rect.center = (width/2, height/2 - b_h/2 - spacing)

    button_1_label = number_font.render("1", True, BLACK)
    button_1_rect = button_1_label.get_rect()
    button_1_rect.center = button_1.center

    button_2_label = number_font.render("2", True, BLACK)
    button_2_rect = button_2_label.get_rect()
    button_2_rect.center = button_2.center

    cancel_button_label = font.render("Cancel", True, BLACK)
    cancel_button_rect = cancel_button_label.get_rect()
    cancel_button_rect.center = cancel_button.center

    confirm_button_label = font.render("Confirm", True, BLACK)
    confirm_button_rect = confirm_button_label.get_rect()
    confirm_button_rect.center = confirm_button.center

    title_font = pygame.font.Font(None, 45)
    font = pygame.font.Font(None, 35)
    number_font = pygame.font.Font(None, 50)

    # Stores states of buttons
    current_selection = None
    cancel_pressed = False
    confirm_pressed = False
    selecting = True
    
    def displayWindow(self, screen):
        # Draw the buttons
        buttonColors = [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY]
        if current_selection == 1:
            buttonColors[0] = GREY
        elif current_selection == 2:
            buttonColors[1] = GREY
        
        if cancel_pressed:
            buttonColors[2] = GREY
        if confirm_pressed:
            buttonColors[3] = GREY
        
        pygame.draw.rect(screen, buttonColors[0], button_1)
        pygame.draw.rect(screen, BLACK, button_1, 2)
        pygame.draw.rect(screen, buttonColors[1], button_2)
        pygame.draw.rect(screen, BLACK, button_2, 2)
        pygame.draw.rect(screen, buttonColors[2], cancel_button)
        pygame.draw.rect(screen, BLACK, cancel_button, 2)
        current_selection and pygame.draw.rect(screen, buttonColors[3], confirm_button)
        current_selection and pygame.draw.rect(screen, BLACK, confirm_button, 2)
        
        # Draw the button labels
        screen.blit(button_1_label, button_1_rect)
        screen.blit(button_2_label, button_2_rect)
        screen.blit(cancel_button_label, cancel_button_rect)
        current_selection and screen.blit(confirm_button_label, confirm_button_rect)
        
        # Draw the title
        screen.blit(title, title_rect)

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

    screen.fill(WHITE)
    pygame.display.flip()

    windowOpen = True
    openWindow = BUTTON_WINDOW

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
                    windowOpen = False
        
        if (openWindow == BUTTON_WINDOW):
            
        else:
            font = pygame.font.SysFont('Calibri', 30, True, False)
            frame = video_getter.frame
            # frame = pose_getter.overlay_frame
            frame = cv2.resize(frame, (1024, 600))

            video_surf = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")

            pygame.draw.rect(video_surf, GREY, [0, 0, 100, 200])
            text = font.render(f"Calories: {pose_getter.read_calories()}", True, BLACK)
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
    pygame.quit()

if __name__ == '__main__':
    main()