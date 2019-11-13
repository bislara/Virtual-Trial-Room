#!C:\Users\LARA\AppData\Local\Programs\Python\Python37-32\python.exe

print("Content-Type: text/html\n")
print ("Hello Python Web Browser!! This is cool!!<img src='images/room_1.jpg'>")


# print("<img src='home_slider.jpg'>")
import numpy as np
import cv2

cap = cv2.VideoCapture('test.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()