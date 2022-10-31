# video_analyzer
demo of face recognition video analytic tool

## Dependecies:
Face Detector - https://github.com/biubug6/Pytorch_Retinaface
Occlusion Awareness Face Recognition Feature - https://github.com/haibo-qiu/FROM
(should be located in src/)

place their weights under weights directory:
recommendation: 
mobilenet0.25_Final.pth - https://drive.google.com/file/d/15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1/view?usp=share_link
model_p5_w1_9938_9470_6503.pth.tar - https://drive.google.com/file/d/1vVRfbA6ANqihtQme__9pJJLna1AiG1PR/view?usp=share_link

## run:
python3 src/main.py 2022-06-12_09-01-39_camera7.mp4 0.1

## Algorithm:
Multi stages video analytic framework:
  Detection stage:
  - Detection:
      RetinaFace is SOTA in face recognition (based on https://paperswithcode.com/task/face-detection)
  - Tracker: 
      SORT (simple-online realtime tracker - https://arxiv.org/abs/1602.00763 IoU based tracker)
  - Face Scoring (and Saving): 
      Using occlusion awareness face recognition feature extracture, use in cosine similarity between the mask and non-mask features for scoring
      based on https://arxiv.org/abs/2112.04016
 Bluring stage:
   - Face blurring based on simple LPF cv2-implementation
   - VideoSaver
   
## TODO:
  complete the dockerization process
    