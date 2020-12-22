import cv2
import argparse
import numpy as np
import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
from threading import Thread



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

def draw_bounding_box(matrix,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
	label = str(classes[class_id]) + " " + str(round(confidence,2))
	color = COLORS[class_id]
	x_new = int((x + x_plus_w)/2)
	p = (x_new,y_plus_h)
	px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
	py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
	p_after = (int(px), int(py))
	img = cv2.circle(img, (p_after[0],p_after[1]), 10 , (0,0,255) , -1)


offline = 1

scale = 0.00392
net = cv2.dnn.readNet(weights, cfg)

def help_function(image,mat,shap):
	Width = image.shape[1]
	Height = image.shape[0]

	blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	outs = net.forward(get_output_layers(net))
	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4

	# for each detetion from each output layer 
	# get the confidence, class id, bounding box params
	# and ignore weak detections (confidence < 0.5)
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

	# go through the detections remaining
	# after nms and draw bounding box
	# print(len(indices))
	image = cv2.warpPerspective(image,mat,shap)
	for i in indices:
		i = i[0]
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]

		draw_bounding_box(mat,image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

	# display output image    
	# cv2.namedWindow("output",cv2.WINDOW_NORMAL) 

	# image = cv2.warpPerspective(image,mat,(1350, 650))
	# result1 = cv2.warpPerspective(cap1,matrix1,shap)
	# cv2.imshow(image)
	image = cv2.resize(image, shap)
	return image

# !rm output.avi


vid=cv2.VideoCapture("videos/raw/lib_trim.mp4")
vid1=cv2.VideoCapture("videos/raw/main_trim.mp4")
vid2=cv2.VideoCapture("videos/raw/redc_trim.mp4")

pts1=np.float32(
	[[1271, 1044],
 [ 146  ,582],
 [1035 , 414],
 [1860,  519],
 # [1271 1044]
 ]
 )

pts2=np.float32(
	[[1145,  270],
 [1145 , 784],
 [ 562,  784],
 [ 564 , 272],
 # [1145  270]
 ]
 )

to_change=np.float32(
	[[1777 , 644],
 [ 777 ,1037],
 [ 136 , 701],
 [1101 , 552],
 # [1777  644]
 ]
 )

project_on = np.float32(
	[[ 819,  956],
 [ 819  ,377],
 [1233 , 377],
 [1231,  954],
 # [ 819  956]
 ]
 )



pts21=np.float32(
	[[1428 ,1000],
 [1902  ,772],
 [1114 , 674],
 [ 502,  780],
 ]
 )

pts22=np.float32(
	[[ 991 , 701],
 [1402  ,701],
 [1402 , 289],
 [ 993,  287],
 ]
 )

shap = (1500, 1000)
scale_percent = 65
width = int(shap[0] * scale_percent / 100)
height = int(shap[1] * scale_percent / 100)
dim = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 30, shap)


matrix1 = cv2.getPerspectiveTransform(pts1,pts2)
matrix = cv2.findHomography(to_change,project_on)[0]
matrix2 = cv2.getPerspectiveTransform(pts21,pts22)

detect = int(input("Enter 1 for top view\nEnter 2 for top view with detection\n"))

counter = 1
while 1:
    d,cap = vid.read()
    d1,cap1 = vid1.read()
    d2,cap2 = vid2.read()
    # print("Frame ",counter)
    counter += 1
    # if counter == 10:
    #     print("Done")
    #     break
    if d != 0 and d1 != 0 and d2 != 0 :
        if(detect == 1):
            # cap = cv2.rotate(cap, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            result1=cv2.warpPerspective(cap1,matrix1,shap)
            result=cv2.warpPerspective(cap,matrix,shap)
            result2=cv2.warpPerspective(cap2,matrix2,shap)

        elif(detect == 2):
            result1 = help_function(cap1,matrix1,shap)
            result = help_function(cap,matrix,shap)
            result2 = help_function(cap2,matrix2,shap)
            
        # cv2.imshow(result2)
        # break
        result1 = result1/255
        result = result/255
        result2 = result2/255
        final = np.where( (result != 0) & (result1 != 0), ((result + result1) /2),result + result1  )
        final = np.where( (final != 0) & (result2 != 0), ((final + result2) /2), final + result2 )
        final = cv2.resize(final, dim, interpolation = cv2.INTER_AREA)
        final = final*255
        # print("LEtsSE")
        final = final.astype( np.uint8)
        out.write(final)
        # resize image
        cv2.imshow("Top_view_labeled",final)
        # break
        key=cv2.waitKey(1)
        if key == 27:
            break
    else:
        print("Could not read Video")
        break

vid1.release()
vid.release()
vid.release()
out.release()
cv2.destroyAllWindows()

# def download(path):
#   from google.colab import files
#   files.download(path)

# download("output.avi")