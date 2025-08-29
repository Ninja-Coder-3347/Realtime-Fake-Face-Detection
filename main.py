from ultralytics import YOLO#Library for loading and using YOLO models for object detection.
import cv2#OpenCV, a library for image and video processing.
import cvzone#A library built on OpenCV that simplifies some common computer vision tasks (e.g., adding text, rectangles).
import math#Provides mathematical operations
import time#Used for calculating frame timing and FPS (frames per second)


# Initialize video capture (for webcam or video file)
# Uncomment one of the following lines based on your input source
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)#Sets the width of the video feed to 640 pixels.
cap.set(4, 480)#Sets the height of the video feed to 480 pixels.
#cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video file

# Load the YOLO model
model = YOLO("../models/best_250.pt")#it download the yolo8 in model folder



# Class names for objects that can be detected
classNames = ['fake', 'real']
print(model.names)

# Initialize frame timing variables
prev_frame_time = 0#Used to calculate the FPS by measuring the time difference between prev fram and new frame
new_frame_time = 0

# Process frames from the video stream
while True:
    new_frame_time = time.time()#record video
    success, img = cap.read()#capture frames
    if not success:
        break

    # Perform detection
    # img-cuurentframe,stream-process result in streaming realtime,verbose-deatiled output log
    results = model(img, stream=True,verbose=False)#verbose-answer givesin black color fonr of text
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box(left top corner and right bottom corner)
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Draws a rectangle with rounded corners around the detected object.
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])#Class index of the detected object.
            if 0 <= cls < len(classNames):
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            else:
                print(f"Invalid class index: {cls}")

            # Put the text of the class name and confidence value on the image
            #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            # Calculate the FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    # Show the image
    cv2.imshow("Image", img)
    # Wait for a key press
    cv2.waitKey(1)


