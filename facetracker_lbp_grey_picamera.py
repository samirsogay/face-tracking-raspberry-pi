#!/usr/bin/env python3

import cv2, sys, time, os
from pantilthat import *
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils


# Frame Size. Smaller is faster, but less accurate.
# Wide and short is better, since moving your head
# vertically is kinda hard!
FRAME_W = 640
FRAME_H = 480

# Default Pan/Tilt for the camera in degrees.
# Camera range is from -90 to 90
cam_pan = 90
cam_tilt = 60

# Set up the CascadeClassifier for face tracking
cascPath = '/usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Turn the camera to the default position
pan(cam_pan-90)
tilt(cam_tilt-90)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.vflip = True
camera.framerate = 25
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)
lastTime = time.time()*1000.0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
    frame = frame.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.1, 3, 0, (10, 10))
    print (time.time()*1000.0-lastTime)
    print (" Found {0} faces!".format(len(faces)))
    lastTime = time.time()*1000.0
    # Draw a rectangle around the faces
    


    for (x, y, w, h) in faces:
        # Draw a green rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Track first face
        
        # Get the center of the face
        x = x + (w/2)
        y = y + (h/2)

        # Correct relative to center of image
        turn_x  = float(x - (FRAME_W/2))
        turn_y  = float(y - (FRAME_H/2))

        # Convert to percentage offset
        turn_x  /= float(FRAME_W/2)
        turn_y  /= float(FRAME_H/2)

        # Scale offset to degrees
        turn_x   *= 2.5 # VFOV
        turn_y   *= 2.5 # HFOV
        cam_pan  += turn_x
        cam_tilt += turn_y

        print(cam_pan-90, cam_tilt-90)

        # Clamp Pan/Tilt to 0 to 180 degrees
        cam_pan = max(0,min(180,cam_pan))
        cam_tilt = max(0,min(180,cam_tilt))

        # Update the servos
        pan(int(cam_pan-90))
        tilt(int(cam_tilt-90))

        break

    frame = cv2.resize(frame, (640,480))
#    frame = cv2.flip(frame, 1)
   
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
        # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
