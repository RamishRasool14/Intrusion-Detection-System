import cv2
import argparse
import numpy as np
import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
from threading import Thread
import ctypes

parser = argparse.ArgumentParser(description = "Path to video")
parser.add_argument('-p', '--path', required=False, help = 'path to input video')
args = parser.parse_args()
# print(args)

cfg = "files/config.cfg"
names = "files/classes"
weights = "files/weights"

classes = None
with open(names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id]) + " " + str(round(confidence,2))

    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 4)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.75, color, 4)



ip = [0,"http://10.130.0.188:8080/video", "http://192.168.8.117:8080/video"]

offline = int(input("Enter 1 for offline or 0 for live "))

if(offline==0):
	cameras = int(input("Enter Number of Live Feeds: "))

	for i in range(0,cameras):
		print("Enter 1 to initialize IP address of camera",i+1,"else 0" )
		if int(input()):
			inp = input("Address(including port): ")
			ip[i] = "https://" + inp + "/video"


scale = 0.00392
net = cv2.dnn.readNet(weights, cfg)

def function(i):
	if(offline == 0):
		video = cv2.VideoCapture(i)
	else:
		if args.path is None:
			args.path = input("Path to video: ")
		video = cv2.VideoCapture(args.path)

	while True:
		status, image = video.read()
		if(status == 0):
			continue
		Width = image.shape[1]
		Height = image.shape[0]

		blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(get_output_layers(net))
		class_ids = []
		confidences = []
		boxes = []
		conf_threshold = 0.5
		nms_threshold = 0.4
		for out in outs:
		    for detection in out:
		        scores = detection[5:]
		        class_id = np.argmax(scores)
		        confidence = scores[class_id]
		        if confidence > 0.5:
		            center_x = int(detection[0] * Width)
		            center_y = int(detection[1] * Height)
		            w = int(detection[2] * Width)
		            h = int(detection[3] * Height)
		            x = center_x - w / 2
		            y = center_y - h / 2
		            class_ids.append(class_id)
		            confidences.append(float(confidence))
		            boxes.append([x, y, w, h])
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

		for i in indices:
		    i = i[0]
		    box = boxes[i]
		    x = box[0]
		    y = box[1]
		    w = box[2]
		    h = box[3]
		    
		    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

		image = cv2.resize(image, (1350,650))
		cv2.imshow("object detection", image)

		key = cv2.waitKey(1)
		if key == 27:
			break

	video.release()

if offline == 1:
	thread1 = Thread(target = function, args = (ip[0],))
	thread1.start()
	thread1.join()
else:
	if(cameras==1):
		thread1 = Thread(target = function, args = (ip[0],))
		
	elif(cameras==2):
		thread1 = Thread(target = function, args = (ip[0],))
		thread2 = Thread(target = function, args = (ip[1],))

	elif(cameras==3):
		thread1 = Thread(target = function, args = (ip[0],))
		thread2 = Thread(target = function, args = (ip[1],))
		thread3 = Thread(target = function, args = (ip[2],))

	if(cameras==1):
		thread1.start()
		
	elif(cameras==2):
		thread1.start()
		thread2.start()

	elif(cameras==3):	
		thread1.start()
		thread2.start()
		thread3.start()


	if(cameras==1):
		thread1.join()
		
	elif(cameras==2):
		thread1.join()
		thread2.join()

	elif(cameras==3):	
		thread1.join()
		thread2.join()
		thread3.join()


cv2.destroyAllWindows()
	
