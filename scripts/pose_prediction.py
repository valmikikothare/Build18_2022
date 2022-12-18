import numpy as np
import torch
import cv2

class PoseEstimator:
    def __init__(self):
        # Specify the paths for the 2 files
        protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
        
        # Read the network into Memory
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    def pose_estimate(self, frame):
        # Specify the input image dimensions
        inWidth = 368
        inHeight = 368
        
        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        
        # Set the prepared object as the input blob of the network
        self.net.setInput(inpBlob)

        out = self.net.forward()

        H = out.shape[2]
        W = out.shape[3]
        # Empty list to store the detected keypoints
        points = []
        for i in range(len()):
            # confidence map of corresponding body's part.
            probMap = out[0, i, :, :]
        
            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
        
            if prob > threshold :
                cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)

if __name__ == '__main__':
    video_file = ''

    cap = cv2.VideoCapture(video_file)
    
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break