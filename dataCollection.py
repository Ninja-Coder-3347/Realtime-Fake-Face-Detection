from cvzone.FaceDetectionModule import FaceDetector
import cv2 #OpenCV, a library for image and video processing.
import cvzone#A library built on OpenCV that simplifies some common computer vision tasks (e.g., adding text, rectangles).
from time import time#Used for calculating frame timing and FPS (frames per second).
#################################################
classID =1 #0 is fake and 1 is real
outputFolderPath='Dataset/DataCollect'
confidence=0.5
save=True  # Enable or disable saving detected data.
blurThreshold = 35 # larger is more focus.# Threshold to classify faces as blurry or not.
debug =False# Display debug information.ji info aplyla disat ahe ti as it is dataset madhye pn store honar

offsetPercentageW =10#offset Additional Padding
offsetPercentageH =20
camWidth, camHeight =640,480
floatingPoint = 6 #folating point nantr kiti digit condider karayche ahet
#################################################

cap = cv2.VideoCapture(0)
cap.set(3,camWidth)# Set width.
cap.set(4,camHeight)# Set height.

detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:

    success, img = cap.read()#Read a frame from the webcam.
    imgOut = img.copy()# Create a copy of the frame for visualization.
    img, bboxs = detector.findFaces(img, draw=False)# Detect faces without drawing boxes.

    listBlur = [] # True false value indicating if the faces are blur or not.
    listInfo = [] #the normalized values and class name for the label text file.

    # Check if any face is detected---->
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            x,y,w,h = bbox["bbox"] # Get bounding box.
            score = bbox["score"][0] #Get detection confidence
            #print(x,y,w,h)
            #score = int(bbox['score'][0] * 100)

            #--------check the score ---------
            if score>confidence:# If confidence exceeds threshold.



                #-------adding an offset to the face Detaected----------
#Adds padding to the bounding box for better detection.

                offsetW =(offsetPercentageW/100)*w
                x=int(x-offsetW)
                w=int(w+offsetW*2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH*3)
                h = int(h + offsetH * 3.5)

                #---------to avoid values below 0 -----------
                # Ensures values are not less than 0.
                if x<0: x=0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # --------Finding the blurriness----------
                imgFace= img[y:y+h,x:x+w] # Crop the detected face.
                cv2.imshow("Face",imgFace)
                # Calculate variance of Laplacian.
                blurValue = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())
                if blurValue>blurThreshold:
                    listBlur.append(True) # Face is not blurry.
                else:
                    listBlur.append(False) # Face is blurry.

# normalization madhye width and height chi vvalue
                # fraction madhye sangitli jaate instead of pixel value---->
                #-------normalize values---------
                ih, iw, _=img.shape# Get image dimensions.
                xc, yc = x+w/2,y+h/2# Calculate center of bounding box.

                xcn, ycn = round(xc/iw,floatingPoint),round(yc/ih,floatingPoint)
                wn, hn = round(w/iw,floatingPoint),round(h/ih,floatingPoint)
                print(xcn, ycn,wn,hn)

                # ---------to avoid values above 1 -----------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1:wn = 1
                if hn > 1: hn = 1

                # Add normalized data.

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}")

                # --------Drawing---------
                # Draw rectangle.

                cv2.rectangle(imgOut,(x,y,w,h),(255,0,0),3)
                cvzone.putTextRect(imgOut,f'Score:{int(score*100)}% Blur:{blurValue}',(x,y-20),scale=1,thickness=2)

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score:{int(score * 100)}% Blur:{blurValue}', (x, y - 20), scale=1,
                                       thickness=2)

                #cvzone.putTextRect(img, f'{score}%', (x, y - 10))
                #cvzone.cornerRect(img, (x, y, w, h))

        #--------to save-------
        if save:
            # Save only if all faces are not blurry.

            if all(listBlur) and listBlur!=[]:
                #----save image-----
                timeNow=time()
                timeNow= str(timeNow).split('.')
                # Save image.

                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)

                #----------save label text file---------->
                for info in listInfo: # Save labels.
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()






    # Display the image in a window named 'Image'
    cv2.imshow("Image", imgOut)
    # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)


