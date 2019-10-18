# Virtual Trial Room
This is done using  3 main techniques:- <br>
<b>1. Pose Estimation<br>
2. Human Parsing<br>
3. Virtual Fitting<br>
</b>

## Prerequisites
Download the following packages for before using the tool
```bash
configobj==5.0.6
flask==1.0.2
h5py==2.8.0
ipython==5.7.0
keras==2.1.6
matplotlib==2.2.2.
numpy==1.14.3
opencv-python==3.4.1.15
pandas==0.23.0
Pillow==5.1.0
pyzmq==17.0.0
scipy==1.1.0
six==1.11.0
tensorflow==1.3.0
```

## Pose Estimation 
Pose estimation is a computer vision technique that detects human figures in images and videos, 
so that one could determine the pose of a person, for example, where someone’s elbow shows up in an image.It is the localization of human joints (also called keypoints - elbows, wrists, etc.)
in images or videos. It is also defined as the search for a specific pose in the space of all articulated poses.
This is done using a pretrained model. <br><br>
<img src="https://github.com/bislara/Virtual-Trial-Room/blob/master/trial%20room/readme_images/1.PNG" >
<img src="https://github.com/bislara/Virtual-Trial-Room/blob/master/trial%20room/readme_images/result1.png">

2D Pose Estimation - It estimate a 2D pose (x,y) coordinates for each joint from a RGB image. It will total 15 2D coordinates.<br>
Human Pose Estimation has some pretty cool applications and is heavily used in Action recognition, Animation, Gaming, etc.

## Human Parcing
Human parsing is the task of segmenting a human image into different fine-grained semantic parts such as head, torso, arms and legs.
It is also done with the help of a pre-trained model. It divides the body into different parts.
<br>
<br>
<img src="https://github.com/bislara/Virtual-Trial-Room/blob/master/trial%20room/readme_images/example_person_vis1.png">

## Virtual Fitting
This is the method in which the cloth is put on the input image or video to get the output in
which the cloth is put on the person to show how the person looks with it. It shows whether the shirt size is 
big or small or perfect to the person's body. Overall it gives an idea how the person will look when he/she wears the cloth. <br>
<img src="https://github.com/bislara/Virtual-Trial-Room/blob/master/trial%20room/readme_images/2.PNG"><br>

<b>Download the [output_video](https://github.com/bislara/Virtual-Trial-Room/blob/master/trial%20room/output_video.mp4) to see the output from our code.</b>
