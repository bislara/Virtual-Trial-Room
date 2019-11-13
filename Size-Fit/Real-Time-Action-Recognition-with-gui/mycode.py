# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:41:35 2019

@author: ASUS
"""
import cv2
import numpy as np
import settings
from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from imutils.video import VideoStream

poseEstimator = None

poseEstimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))

cap=cv2.VideoCapture(0)           
   
#cap = VideoStream(src='rtsp://admin:rashmi2018@192.168.51.162/PSIA/streaming/channels/102').start()

while True:
    
    ret,frame=cap.read()
    ret=True
    if ret :
        
        show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
        
        humans = poseEstimator.inference(show)
                   
        show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
        cv2.imshow("frame",show)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
        