#
# Beispielcode zur Extraktion von einzelnen Frames mit OpenCV
#

import cv2
cap = cv2.VideoCapture('./dash_cam.mov') #Alternativ auch ein MP4-Video
frame_index = 0
while(cap.isOpened()):
    _, frame = cap.read()
    cv2.imshow('frame',frame)
    cv2.imwrite("frame_" + str(frame_index) + ".jpg", frame)
    frame_index = frame_index +1 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
