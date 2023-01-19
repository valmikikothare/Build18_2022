# Credit for some implementation of tflite model: https://github.com/ecd1012/rpi_pose_estimation/blob/main/

import numpy as np
import cv2
import time
import os
import argparse
import tflite_runtime as tf
from tflite_runtime.interpreter import Interpreter

class PoseEstimator:
    def __init__(self, model_path):
        # Specify the paths for the 2 files
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.output_stride = 4
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        def mod(a, b):
            """find a % b"""
            floored = np.floor_divide(a, b)
            return np.subtract(a, np.multiply(floored, b))

        def sigmoid(x):
            """apply sigmoid actiation to numpy array"""
            return 1/ (1 + np.exp(-x))
            
        def sigmoid_and_argmax2d(inputs, threshold):
            """return y,x coordinates from heatmap"""
            #v1 is 9x9x17 heatmap
            v1 = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            height = v1.shape[0]
            width = v1.shape[1]
            depth = v1.shape[2]
            reshaped = np.reshape(v1, [height * width, depth])
            reshaped = sigmoid(reshaped)
            #apply threshold
            reshaped = (reshaped > threshold) * reshaped
            coords = np.argmax(reshaped, axis=0)
            yCoords = np.round(np.expand_dims(np.divide(coords, width), 1)) 
            xCoords = np.expand_dims(mod(coords, width), 1) 
            return np.concatenate([yCoords, xCoords], 1)

        def get_offset_point(y, x, offsets, keypoint, num_key_points):
            """get offset vector from coordinate"""
            y_off = offsets[y,x, keypoint]
            x_off = offsets[y,x, keypoint+num_key_points]
            return np.array([y_off, x_off])
            

        def get_offsets(output_details, coords, num_key_points=17):
            """get offset vectors from all coordinates"""
            offsets = interpreter.get_tensor(output_details[1]['index'])[0]
            offset_vectors = np.array([]).reshape(-1,2)
            for i in range(len(coords)):
                heatmap_y = int(coords[i][0])
                heatmap_x = int(coords[i][1])
                #make sure indices aren't out of range
                if heatmap_y >8:
                    heatmap_y = heatmap_y -1
                if heatmap_x > 8:
                    heatmap_x = heatmap_x -1
                offset_vectors = np.vstack((offset_vectors, get_offset_point(heatmap_y, heatmap_x, offsets, i, num_key_points)))  
            return offset_vectors

        def draw_lines(keypoints, image, bad_pts):
            """connect important body part keypoints with lines"""
            #color = (255, 0, 0)
            color = (0, 255, 0)
            thickness = 2
            #refernce for keypoint indexing: https://www.tensorflow.org/lite/models/pose_estimation/overview
            body_map = [[5,6], [5,7], [7,9], [5,11], [6,8], [8,10], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]]
            for map_pair in body_map:
                #print(f'Map pair {map_pair}')
                if map_pair[0] in bad_pts or map_pair[1] in bad_pts:
                    continue
                start_pos = (int(keypoints[map_pair[0]][1]), int(keypoints[map_pair[0]][0]))
                end_pos = (int(keypoints[map_pair[1]][1]), int(keypoints[map_pair[1]][0]))
                image = cv2.line(image, start_pos, end_pos, color, thickness)
            return image


            #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
            while True:
                print('running loop')
                # Start timer (for calculating frame rate)
                t1 = cv2.getTickCount()
                
                # Grab frame from video stream
                frame1 = videostream.read()
                # Acquire frame and resize to expected shape [1xHxWx3]
                frame = frame1.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)
                
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()
                
                #get y,x positions from heatmap
                coords = sigmoid_and_argmax2d(output_details, min_conf_threshold)
                #keep track of keypoints that don't meet threshold
                drop_pts = list(np.unique(np.where(coords ==0)[0]))
                #get offets from postions
                offset_vectors = get_offsets(output_details, coords)
                #use stide to get coordinates in image coordinates
                keypoint_positions = coords * output_stride + offset_vectors
            
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(keypoint_positions)):
                    #don't draw low confidence points
                    if i in drop_pts:
                        continue
                    # Center coordinates
                    x = int(keypoint_positions[i][1])
                    y = int(keypoint_positions[i][0])
                    center_coordinates = (x, y)
                    radius = 2
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.circle(frame_resized, center_coordinates, radius, color, thickness)
                    if debug:
                        cv2.putText(frame_resized, str(i), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1) # Draw label text
     
                frame_resized = draw_lines(keypoint_positions, frame_resized, drop_pts)

                # Draw framerate in corner of frame - remove for small image display
                #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                #cv2.putText(frame_resized,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

                # Calculate framerate
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc= 1/time1
                f.append(frame_rate_calc)
    
                #save image with time stamp to directory
                path = str(outdir) + '/'  + str(datetime.datetime.now()) + ".jpg"

                status = cv2.imwrite(path, frame_resized)

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q') or led_on and not GPIO.input(17):
                    print(f"Saved images to: {outdir}")
                    GPIO.output(4, False)
                    led_on = False
                    # Clean up
                    cv2.destroyAllWindows()
                    videostream.stop()
                    time.sleep(2)
                    break

    def estimate(self, frame, threshold=0.1):
        frame_height, frame_width = frame.shape[:2]
        # Specify the input image dimensions
        input_image = tf.cast(frame, dtype=tf.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())

        self.interpreter.invoke()

        # Output is a [1, 1, 17, 3] numpy array.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # assert out.shape[1] == self.nPoints
        # Empty list to store the detected keypoints
        points = []
        for i in range(self.n_points):
            # confidence map of corresponding body's part.
            prob_map = out[0, i, :, :]
        
            # Find global maxima of the probMap.
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        
            # Scale the point to fit on the original image
            x = (frame_width * point[0]) / W
            y = (frame_height * point[1]) / H
        
            if prob > threshold :
                # Add the point to the list if the probability is greater than the threshold
                assert x >= 0 and y >= 0
                points.append((int(x), int(y)))
            else :
                points.append((-1, -1))

        return np.array(points)

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

    pose = PoseEstimator(mode='mpi')
    
    while True:
        t = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code) 

        points = pose.estimate(frame)

        # Draw Skeleton
        for pair in pose.pose_pairs:
            partA = pair[0]
            partB = pair[1]

            if (points[partA,:] >= 0).all() and (points[partB,:] >= 0).all():
                cv2.line(frame, points[partA,:], points[partB,:], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB,:], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        
        # for i, point in enumerate(points):
        #     # cv2.circle(frame, point, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        #     cv2.putText(frame, str(i), point, cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        
        vid_writer.write(frame)

        cv2.imshow('Output Pose', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    vid_writer.release()

if __name__ == '__main__':
    main()
