import numpy as np
import cv2 

img = cv2.imread("people.jpg",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
path1 = "lbpcascade_profileface.xml"
path2 = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(path1)
face_cascade = cv2.CascadeClassifier(path2)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40,40))

print (len(faces))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()