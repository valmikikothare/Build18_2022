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
        self.t = time.time()
        self.total_calories = 0
        self.overlay_frame = None
        self.in_frame = False
        self.rotate_code = rotate_code
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
            
            calories = 0
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
                    calories = self.calorie.estimate(displacement, dt)
                    self.window_x.pop(0)
                    self.window_y.pop(0)
                    self.window_t.pop(0)
            
            self.overlay_frame = frame
            self.total_calories += calories
            self.in_frame = valid.all()

    def stop(self):
        self.stopped = True

    def read(self):
        return self.overlay_frame, self.total_calories, self.in_frame

class ButtonWindow():
    def __init__(self):
        # Create the buttons
        width, height = (1024, 600)
        b_w = width/4.5
        b_h = height/10
        spacing = b_w/3
        self.button_1 = pygame.Rect(width/2 - b_w/2 - spacing - b_w, height/2-b_h/2, b_w, b_h)
        self.button_2 = pygame.Rect(width/2 - b_w/2, height/2-b_h/2, b_w, b_h)
        self.cancel_button = pygame.Rect(width/2 + b_w/2 + spacing, height/2-b_h/2, b_w, b_h)
        self.confirm_button = pygame.Rect(width/2 - b_w/2, height/2+b_h/2+spacing, b_w, b_h)

        title_font = pygame.font.SysFont('Calibri', 45, True, False)
        font = pygame.font.SysFont('Calibri', 35, True, False)
        number_font = pygame.font.SysFont('Calibri', 50, True, False)

        # Selection Menu Title
        self.title = title_font.render("Please select your desired snack :)", True, BLACK)
        self.title_rect = self.title.get_rect()
        self.title_rect.center = (width/2, height/2 - b_h/2 - spacing)

        self.button_1_label = number_font.render("1", True, BLACK)
        self.button_1_rect = self.button_1_label.get_rect()
        self.button_1_rect.center = self.button_1.center

        self.button_2_label = number_font.render("2", True, BLACK)
        self.button_2_rect = self.button_2_label.get_rect()
        self.button_2_rect.center = self.button_2.center

        self.cancel_button_label = font.render("Cancel", True, BLACK)
        self.cancel_button_rect = self.cancel_button_label.get_rect()
        self.cancel_button_rect.center = self.cancel_button.center

        self.confirm_button_label = font.render("Confirm", True, BLACK)
        self.confirm_button_rect = self.confirm_button_label.get_rect()
        self.confirm_button_rect.center = self.confirm_button.center

        # Stores states of buttons
        self.current_selection = None
        self.cancel_pressed = False
        self.confirm_pressed = False
        self.selecting = True
    
    def setCancelPressed(self, input):
        self.cancel_pressed = input

    def setCurrentSelection(self, input):
        self.current_selection = input

    def setConfirmPressed(self, input):
        self.confirm_pressed = input

    def setSelecting(self, input):
        self.selecting = input

    def readCurrentSelection(self):
        return self.current_selection

    def readButtons(self):
        return self.button_1, self.button_2, self.cancel_button, self.confirm_button
    
    def displayWindow(self, screen):
        # Draw the buttons
        buttonColors = [LIGHT_GREY, LIGHT_GREY, LIGHT_GREY, LIGHT_GREY]
        if self.current_selection == 1:
            buttonColors[0] = GREY
        elif self.current_selection == 2:
            buttonColors[1] = GREY
        
        if self.cancel_pressed:
            buttonColors[2] = GREY
        if self.confirm_pressed:
            buttonColors[3] = GREY
        
        pygame.draw.rect(screen, buttonColors[0], self.button_1)
        pygame.draw.rect(screen, BLACK, self.button_1, 2)
        pygame.draw.rect(screen, buttonColors[1], self.button_2)
        pygame.draw.rect(screen, BLACK, self.button_2, 2)
        pygame.draw.rect(screen, buttonColors[2], self.cancel_button)
        pygame.draw.rect(screen, BLACK, self.cancel_button, 2)
        self.current_selection and pygame.draw.rect(screen, buttonColors[3], self.confirm_button)
        self.current_selection and pygame.draw.rect(screen, BLACK, self.confirm_button, 2)
        
        # Draw the button labels
        screen.blit(self.button_1_label, self.button_1_rect)
        screen.blit(self.button_2_label, self.button_2_rect)
        screen.blit(self.cancel_button_label, self.cancel_button_rect)
        self.current_selection and screen.blit(self.confirm_button_label, self.confirm_button_rect)
        
        # Draw the title
        screen.blit(self.title, self.title_rect)

        pygame.display.flip()

class VideoWindow():
    def __init__(self, video_getter, pose_getter):
        self.font = pygame.font.SysFont('Calibri', 30, True, False)
        self.video_getter = video_getter
        self.pose_getter = pose_getter
        self.calorie_display = pygame.Rect(0, 0, 100, 200)
        self.discount_display = pygame.Rect(0, 200, 100, 200)
        self.stop_display = pygame.Rect(0, 400, 100, 200)

    def displayWindow(self, screen, STOP_COLOR, render_pose=False):
        if render_pose:
            frame, total_calories, in_frame = self.pose_getter.read()
        else:
            frame = self.video_getter.read()
            _, total_calories, in_frame = self.pose_getter.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1024, 600))

        video_surf = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")

        pygame.draw.rect(video_surf, GREY, self.calorie_display)
        calorie_text = self.font.render(f"Calories: {total_calories:.2f}", True, BLACK)
        calorie_text = pygame.transform.rotate(calorie_text, 270)
        calorie_text_rect = calorie_text.get_rect()
        calorie_text_rect.center = self.calorie_display.center
        video_surf.blit(calorie_text, calorie_text_rect)

        pygame.draw.rect(video_surf, GREEN, self.discount_display)
        display_text = self.font.render(f"Discount: TODO", True, BLACK)
        display_text = pygame.transform.rotate(display_text, 270)
        display_text_rect = display_text.get_rect()
        display_text_rect.center = self.discount_display.center
        video_surf.blit(display_text, display_text_rect)

        pygame.draw.rect(video_surf, STOP_COLOR, self.stop_display)
        stop_text = self.font.render("Pay", True, BLACK)
        stop_text = pygame.transform.rotate(stop_text, 270)
        stop_text_rect = stop_text.get_rect()
        stop_text_rect.center = self.stop_display.center
        video_surf.blit(stop_text, stop_text_rect)

        screen.blit(video_surf, (0, 0))

        pygame.display.flip()
    
    def readStop(self):
        return self.stop_display

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
    buttonWindow = ButtonWindow()
    videoWindow = VideoWindow(video_getter, pose_getter)

    while windowOpen:
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                windowOpen = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (openWindow == BUTTON_WINDOW):
                    current_selection = buttonWindow.readCurrentSelection()
                    button_1, button_2, cancel_button, confirm_button = buttonWindow.readButtons()
                    if current_selection != 1 and button_1.collidepoint(event.pos):
                        buttonWindow.setCurrentSelection(1)
                    elif current_selection != 2 and button_2.collidepoint(event.pos):
                        buttonWindow.setCurrentSelection(2)
                    elif cancel_button.collidepoint(event.pos):
                        buttonWindow.setCancelPressed(True)
                        buttonWindow.setCurrentSelection(None)
                    elif current_selection and confirm_button.collidepoint(event.pos):
                        buttonWindow.setConfirmPressed(True)
                elif (openWindow == VIDEO_WINDOW):
                    stop_display = videoWindow.readStop()
                    if stop_display.collidepoint(event.pos):
                        STOP_COLOR = DARK_RED
            elif event.type == pygame.MOUSEBUTTONUP:
                if (openWindow == BUTTON_WINDOW):
                    button_1, button_2, cancel_button, confirm_button = buttonWindow.readButtons()
                    if cancel_button.collidepoint(event.pos):
                        buttonWindow.setCancelPressed(False)
                    elif current_selection and confirm_button.collidepoint(event.pos):
                        buttonWindow.setConfirmPressed(False)
                        buttonWindow.setSelecting(False)
                        openWindow = VIDEO_WINDOW
                elif (openWindow == VIDEO_WINDOW):
                    stop_display = videoWindow.readStop()
                    if stop_display.collidepoint(event.pos):
                        windowOpen = False
                        openWindow = BUTTON_WINDOW
        
        if (openWindow == BUTTON_WINDOW):
            buttonWindow.displayWindow(screen)
        else:
            videoWindow.displayWindow(screen, STOP_COLOR)

    # vid_writer.release()
    pygame.quit()

if __name__ == '__main__':
    main()