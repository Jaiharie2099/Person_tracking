
# python object_tracking_social_distancing.py --prototxt frozen_inference_graph.pb --model graph.pbtxt --video vtest.avi
from addfiles.centroidtracker import CentroidTracker
from imutils.video import FileVideoStream
from scipy.spatial import distance
import numpy as np
import argparse
import imutils
import time
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", type=str, default="frozen_inference_graph.pb",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str, default="graph.pbtxt",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", type=str, default="vtest.avi",
	help="path to input video file")
args = vars(ap.parse_args())
ct = CentroidTracker()
(H, W) = (None, None)
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow(args["prototxt"], args["model"])
print("[INFO] starting video stream...")
vs = FileVideoStream(args["video"]).start()
time.sleep(2.0)
i=0
while(i==0):
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    height = (np.shape(frame))[0]
    i+=1
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (400,height))
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    if W is None or H is None:
        (H, W) = frame.shape[:2]    
    centroids=[]
    distdata=[]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []   
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
            (startX, startY, endX, endY) = box.astype("int")
            centroids.append(startX+(int((endX-startX)/2), startY+(int((endY-startY)/2))))
            cv2.rectangle(frame, (startX,startY), (endX, endY), (0,255,0), 2)
    objects = ct.update(rects)    
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)   
    for i in range(len(centroids)-1):
        for j in range(i+1,len(centroids)):
            distdata.append([i,j,distance.euclidean(centroids[i], centroids[j])])    
    for (i,j,distancee) in distdata:
        if(distancee<50):
            cv2.rectangle(frame,(rects[i][0],rects[i][1]),(rects[i][2], rects[i][3]),(0,0,255),2)
            cv2.rectangle(frame,(rects[j][0],rects[j][1]),(rects[j][2], rects[j][3]),(0,0,255),2)
        else:
            continue
    out.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
