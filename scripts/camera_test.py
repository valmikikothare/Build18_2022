import cv2
import numpy

def main():
    # Webcam Capture
    cap = cv2.VideoCapture(0)
    rotate_code = None
    
    if not cap.isOpened():
        raise Exception("VideoCapture object cannot be opened")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code) 

        cv2.imshow('Output Pose', frame)
        if cv2.waitKey(1) != -1:
            break

if __name__ == '__main__':
    main()
