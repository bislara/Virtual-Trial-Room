# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:41:35 2019

@author: ASUS
"""
import cv2
import numpy as np
import settings
from pose.estimator import TfPoseEstimator
poseEstimator = None
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    
    show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
    show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
    humans = poseEstimator.inference(show)
    print(humans)            
    show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)