# importing modules

import cv2
import numpy as np
import time
from sklearn.preprocessing import normalize

'''
References:

For Object Detection: 

YoloV3 is faster than Any other Object Detection Algorithms.
So,For High Fps we used Yolov3 for our Project.

For Depth Estimation:

Link : https://pysource.com/2019/07/08/yolo-real-time-detection-on-cpu/

'''


class ObstacleAvoidance:
    # for pre-processing purpose
    def _init_(self):
        # loading the yolov3-tiny.weights and its config file
        self.net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

        # this list will contain the list of the all the objects which yolo model can detects
        self.classes = []
        with open('coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()  # convo layers
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # starting time and the frame ID in order to calculate FPS we are processing.
        self.frame_id = 0
        self.starting_time = time.time()

    # for Object detection & Direction of frame (left,right,front)
    def processImage(self, direction):
        # making projection of Drone on our image(cropping original frame into smaller frame)
        self.crop_img = self.img[self.y:self.y + self.h, self.x:self.x + self.w]

        # Blob it’s used to extract feature from the image and to resize them  as we can't
        # img as input directly
        blob = cv2.dnn.blobFromImage(self.crop_img, 1.0 / 255.0, (116, 116), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        # Outs -> is an array that contains all the information about objects detected,
        # their position and the confidence about the detection.
        outs = self.net.forward(self.output_layers)

        class_ids = []  # info of detected objects
        confidences = []  # confidence score 0 - 1
        boxes = []  # contain the coordinates of the rectangle surrounding the object detected.

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)

                # we set a threshold confidence of 0.3, if it’s greater we
                # consider the object correctly detected, otherwise we skip it.
                confidence = scores[class_id]
                if confidence > .3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # When we perform the detection, it happens that we have more boxes for the same object,
        # so we should use another function to remove this “noise”.
        # It’s called Non maximum suppression.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .4, .3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])  # it’s the name of the object detected
                confidence = confidences[i]
                self.color = self.colors[class_ids[i]]  # setting color of rectangles
                cv2.rectangle(self.crop_img, (x, y), (x + w, y + h), self.color, 2)
                cv2.rectangle(self.crop_img, (x, y), (x + w, y + 30), self.color, -1)
                cv2.putText(self.crop_img, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 255, 255), 3)
                # fps calculation
                elapsed_time = time.time() - self.starting_time
                fps = self.frame_id / elapsed_time
                cv2.putText(self.crop_img, "FPS: " + str(round(fps, 2)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        # displaying the image
        cv2.imshow(direction, self.crop_img)
        if len(class_ids) == 0:
            return [1]
        return [0, self.crop_img]

        # (x,y) - top left corners
        # (x+w,y+h) - bottom right corners

    # Function to check Whether there is any object in our front or not
    def goStraight(self):
        self.x = 235
        self.y = 196
        self.h = 88
        self.w = 282
        return self.processImage("Straight")

    # Function to check whether there is any object in our left or not
    def goLeft(self):
        self.x = 0
        self.y = 196
        self.h = 88
        self.w = 235
        return self.processImage("Left")

    # function to check whether there is object in our Right or not
    def goRight(self):
        self.x = 517
        self.y = 234
        self.h = 88
        self.w = 235
        return self.processImage("Right")

    def depthCalculation(self,img,x,y,h,w):

        # Filtering
        kernel = np.ones((3, 3), np.uint8)

        window_size = 3
        min_disp = 2
        num_disp = 130 - min_disp
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=window_size,
                                       uniquenessRatio=10,
                                       speckleWindowSize=100,
                                       speckleRange=32,
                                       disp12MaxDiff=5,
                                       P1=8 * 3 * window_size ** 2,
                                       P2=32 * 3 * window_size ** 2)

        # Used for the filtered image
        stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

        # WLS FILTER Parameters
        lmbda = 80000
        sigma = 1.8
        visual_multiplier = 1.0

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        # Start Reading Camera images

        # Convert from color(BGR) to gray
        grayR = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the Depth_image
        disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
        dispL = disp
        dispR = stereoR.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Using the WLS filter
        filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        # cv2.imshow('Disparity Map', filteredImg)
        disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp

        # Filtering the Results with a closing filter
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,
                                   kernel)

        # Colors map
        dispc = (closing - closing.min()) * 255
        dispC = dispc.astype(np.uint8)
        disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
        filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

        # Show the result for the Depth_image
        # cv2.imshow('Disparity', disp)
        # cv2.imshow('Closing',closing)
        # cv2.imshow('Color Depth',disp_Color)
        #cv2.imshow('Filtered Color Depth', filt_Color)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # get distance

        average = 0
        c = 0
        for u in range(x, x+w):
            for v in range(y, y+h):
                if filteredImg[v][u]>0:
                    average+=filteredImg[v][u]
                    c+=1
        average = average / c
        #Distance = -593.97 * average * (3) + 1506.8 * average * (2) - 1373.1 * average + 522.06
        Distance  = 400*450/average
        #Distance = np.around(Distance * 0.01, decimals=2)
        #print('Distance: ' + str(Distance) + ' m')
        return Distance/1000


# initializing the object of above class
obj = ObstacleAvoidance()
obj.droneW = 800  # drone width (mm)
obj.droneH = 250  # drone Height (mm)
obj.focalLength = 400  # focal length of camera which is used in drone(in pxls)
D = 1000  # min distance from which our drone will start detection  (mm)

# Opening The webcam
cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(1)
obj.frame_id = 0  # counter for number of frames
d = 0
while True:
    _, obj.img = cap.read()  # reading one frame at a time
   # _, obj.img2 = cap2.read()  # reading one frame from second camera
    obj.img = cv2.resize(obj.img, (760, 480))  # resizing it to original resolution of our drone
    obj.img = cv2.flip(obj.img, 1)  # flipping the input image
    obj.frame_id += 1  # increamenting the counter
    height, width, channels = obj.img.shape  # Getting the dimensions of captured frame
    obj.Lframe = obj.img
    obj.Rframe = obj.img   # obj.Rframe = obj.img2   #for right frame in case of stereo camera
    if obj.goStraight() == 1:  # 1 indicates path is clear
        print("Go Straight !")
        d = 0
        obj.goStraight()
    else:
        d = obj.depthCalculation(obj.img, x=235, y=196, h=88, w=282)
        if d>3:
            obj.goStraight()
            print("From Center : ",d)
        else:
            if obj.goLeft() == 1:  # something there in front of so checking for left side
                print("Go Left !")
                obj.goLeft()
            else:
                dl = obj.depthCalculation(obj.img,x = 0,y = 196,h = 88,w = 235)
                if dl>3:
                    obj.goLeft()
                    print("From Left : ",dl)
                else:
                    if obj.goRight()==1:
                        print("Go Right !")
                        obj.goRight()
                    else:
                        dr = obj.depthCalculation(obj.img,x = 517,y = 234,h = 88,w = 235)
                        if dr>3:
                            print("Go Right !")
                            obj.goRight()
                            print("right:",dr)
                        else:
                            print("Move Back/Stop")
    if cv2.waitKey(1) == ord('q'):
        break

# remaining cleaning stuff
cap.release()
cv2.destroyAllWindows()