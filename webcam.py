from time import sleep
import cv2
from computer_vision import *

cv2.namedWindow("Image")
cam = cv2.VideoCapture(0)
cv2.waitKey(0)

if cam.isOpened(): # try to get the first frame
    rval, frame = cam.read()
else:
    rval = False

while rval:
    frame = roberts_cross(frame, 10, False)
    cv2.imshow("Image", frame)
    rval, frame = cam.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("Image")
cam.release()