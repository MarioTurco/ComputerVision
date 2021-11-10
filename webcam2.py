import cv2

from computer_vision import *

#cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
     rval, frame = vc.read()
else:
     rval = False
cv2.imshow("image", frame)
cv2.waitKey(0)
#cv2.imshow("preview", frame)
