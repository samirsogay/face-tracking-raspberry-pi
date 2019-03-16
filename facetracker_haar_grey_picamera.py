from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import sys
import imutils
from pantilthat import *

# Get user supplied values
#cascPath = sys.argv[1]
cascPath = '/home/pi/PanTiltFacetracker/haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Frame Size. Smaller is faster, but less accurate.
# Wide and short is better, since moving your head
# vertically is kinda hard!
FRAME_W = 640
FRAME_H = 480

# Default Pan/Tilt for the camera in degrees.
# Camera range is from -90 to 90
cam_pan = 90
cam_tilt = 60

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
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
    frame = image.array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    )
    print (time.time()*1000.0-lastTime)
    print (" Found {0} faces!".format(len(faces)))
    lastTime = time.time()*1000.0
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # Correct relative to center of image
        turn_x  = float(x - (FRAME_W/2))
        turn_y  = float(y - (FRAME_H/2))

        # Convert to percentage offset
        turn_x  /= float(FRAME_W/2)
        turn_y  /= float(FRAME_H/2)

        # Scale offset to degrees
        turn_x   *= 2.5 # VFOV
        turn_y   *= 2.5 # HFOV
        print (turn_x)
        print (turn_y)
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

    image = cv2.resize(frame, (640,480))
#    image = cv2.flip(frame, -1) 

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
 
        # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        
  
        

