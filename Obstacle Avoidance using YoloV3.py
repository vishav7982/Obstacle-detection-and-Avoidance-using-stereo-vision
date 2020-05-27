# importing modules
import cv2
import numpy as np
import time
'''
References:

For Object Detection: 
YoloV3 is faster than Any other Object Detection Algorithms.
So,For High Fps we used Yolov3 for our Project.
For Depth Estimation:

Link : https://pysource.com/2019/07/08/yolo-real-time-detection-on-cpu/
'''


class ObstacleAvoidance:
    #for pre-processing purpose
    def __init__(self):
        # loading the yolov3-tiny.weights and its config file
        self.net = cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg')

        # this list will contain the list of the all the objects which yolo model can detects
        self.classes = []
        with open('coco.names','r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames() #convo layers
        self.output_layers = [self.layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255,size = (len(self.classes),3))

        # starting time and the frame ID in order to calculate FPS we are processing.
        self.frame_id = 0
        self.starting_time = time.time()

    # for Object detection & Direction of frame (left,right,front)
    def processImage(self,direction):
        # making projection of Drone on our image(cropping original frame into smaller frame)
        crop_img = self.img[self.y:self.y + self.h, self.x:self.x + self.w]

        # Blob it’s used to extract feature from the image and to resize them  as we can't
        # img as input directly
        blob = cv2.dnn.blobFromImage(crop_img, 1.0 / 255.0, (116, 116), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        #Outs -> is an array that contains all the information about objects detected,
        # their position and the confidence about the detection.
        outs = self.net.forward(self.output_layers)

        class_ids = [] #info of detected objects
        confidences = [] #confidence score 0 - 1
        boxes = [] # contain the coordinates of the rectangle surrounding the object detected.

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)

                #we set a threshold confidence of 0.3, if it’s greater we
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

        #When we perform the detection, it happens that we have more boxes for the same object,
        # so we should use another function to remove this “noise”.
        #It’s called Non maximum suppression.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .4, .3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]]) #it’s the name of the object detected
                confidence = confidences[i]
                self.color = self.colors[class_ids[i]] #setting color of rectangles
                cv2.rectangle(crop_img, (x, y), (x + w, y + h), self.color, 2)
                cv2.rectangle(crop_img, (x, y), (x + w, y + 30), self.color, -1)
                cv2.putText(crop_img, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 255, 255), 3)
                # fps calculation
                elapsed_time = time.time() - self.starting_time
                fps = self.frame_id / elapsed_time
                cv2.putText(crop_img, "FPS: " + str(round(fps, 2)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        # displaying the image
        cv2.imshow(direction, crop_img)
        if len(class_ids) == 0:
            return 1
        return 0




        # (x,y) - top left corners
        # (x+w,y+h) - top left corners

# Function to check Whether there is any object in our front or not
    def goStraight(self):
        self.x = 235
        self.y = 196
        self.h = 88
        self.w = 282
        return self.processImage("Straight")

#Function to check whether there is any object in our left or not
    def goLeft(self):
        self.x = 0
        self.y = 196
        self.h = 88
        self.w = 235
        return self.processImage( "Left")

#function to check whether there is object in our Right or not
    def goRight(self):
        self.x = 517
        self.y = 234
        self.h = 88
        self.w = 235
        return self.processImage("Right")


# initializing the object of above class
obj = ObstacleAvoidance()
obj.droneW = 800  # drone width (mm)
obj.droneH = 250  # drone Height (mm)
obj.focalLength = 733  # focal length of camera which is used in drone(in pxls)
D = 1000             # min distance from which our drone will start detection  (mm)

#Opening The webcam
cap = cv2.VideoCapture(0)
obj.frame_id = 0 # counter for number of frames

while True:
    _,obj.img = cap.read()  #reading one frame at a time
    obj.img = cv2.resize(obj.img,(752,480)) #resizing it to original resolution of our drone
    obj.img = cv2.flip(obj.img,1) #flipping the input image
    obj.frame_id += 1   #increamenting the counter
    height,width,channels = obj.img.shape # Getting the dimensions of captured frame

    if obj.goStraight() == 1:    # 1 indicates path is clear
        print("Go Straight !")
        obj.goStraight()

    elif obj.goLeft() == 1: # something there in front off so checking for left side
        print("Go Left !")
        obj.goLeft()

    elif obj.goRight() == 1: #check for Right Frame if it is clear then go in
        print("Go Right !")
        obj.goRight()

    if cv2.waitKey(1) == ord('q'):
        break

#remaining cleaning stuff
cap.release()
cv2.destroyAllWindows()
