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

k = 10
gauss = cv2.getGaussianKernel(k,7)
gauss = gauss*gauss.T
gauss = (gauss/gauss[int(k/2),int(k/2)])
gauss = gauss - 0.3
shap = (1500, 1000)
scale_percent = 75
width = int(shap[0] * scale_percent / 100)
height = int(shap[1] * scale_percent / 100)
dim = (width, height)

check = 1
check = int(input("Press 1 to read points from file (fast) or 0 otherwise (slow) "))

if check == 0:
	file = open("points.txt","w+") 


classes = None
with open(names, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bounding_box(mask,matrix,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + " " + str(round(confidence,2))
    color = COLORS[class_id]
    x_new = int((x + x_plus_w)/2)
    p = (x_new,y_plus_h)
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    p_after = (int(px), int(py))

    if check == 0:
        file.write(str(p_after[0])+","+str(p_after[1])+"/")
    extract = mask[int(p_after[1]-k/2):int(p_after[1]+k/2),int(p_after[0]-k/2):int(p_after[0]+k/2),:]
    if(extract.shape[0:2] == gauss.shape):
        extract = cv2.cvtColor(extract, cv2.COLOR_BGR2GRAY)
        add = (gauss + extract)/2                                               
        add = cv2.applyColorMap( ((add)*255).astype(np.uint8),cv2.COLORMAP_JET).astype(np.float32)/255
        mask[int(p_after[1]-k/2):int(p_after[1]+k/2),int(p_after[0]-k/2):int(p_after[0]+k/2),:] = add

offline = 1

scale = 0.00392
net = cv2.dnn.readNet(weights, cfg)

def help_function(image,mat,mask):
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
	image = cv2.warpPerspective(image,mat,(Width,Height))
	
	for i in indices:
		i = i[0]
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]

		draw_bounding_box(mask,mat,image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
	return image


vid=cv2.VideoCapture("videos/raw/lib_trim.mp4")
vid1=cv2.VideoCapture("videos/raw/main_trim.mp4")
vid2=cv2.VideoCapture("videos/raw/redc_trim.mp4")


pts1=np.float32(
	[[1271, 1044],
 [ 146  ,582],
 [1035 , 414],
 [1860,  519],
 ]
 )

pts2=np.float32(
	[[1145,  270],
 [1145 , 784],
 [ 562,  784],
 [ 564 , 272],
 ]

 )

to_change=np.float32(
	[[1777 , 644],
 [ 777 ,1037],
 [ 136 , 701],
 [1101 , 552],
 ]
 )

project_on = np.float32(
	[[ 819,  956],
 [ 819  ,377],
 [1233 , 377],
 [1231,  954],
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

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 30, dim)


matrix1=cv2.getPerspectiveTransform(pts1,pts2)
matrix = cv2.getPerspectiveTransform(to_change,project_on)
matrix2=cv2.getPerspectiveTransform(pts21,pts22)
frames = 0
frames = int(input("Input the number of frames for animated heatmap (0 for static): "))
count = 0
counter = 1



if check == 1:
	reading = open("videos/points.txt","r")
	data = reading.read()
	data = data.split("Frame")
	vid3 = cv2.VideoCapture("videos/TopView.avi")


number = 1
shap = (1920,1080)
mask = np.zeros((1080, 1920, 3)).astype(np.float32)
while 1:

    if check == 0:
        d,cap = vid.read()
        d1,cap1 = vid1.read()
        d2,cap2 = vid2.read()  
    else:
        d = 1
        d1 = 1
        d2 = 1
    
    print("Frame", counter)
    if check == 0:
    	file.write("Frame")
    counter += 1
    if d != 0 and d1 != 0 and d2 != 0 :

        if check == 0:
            result1 = help_function(cap1,matrix1,mask)
            result = help_function(cap,matrix,mask)
            result2 = help_function(cap2,matrix2,mask)
            result1 = result1/255
            result = result/255
            result2 = result2/255
            final = np.where( (result != 0) & (result1 != 0), ((result + result1) /2),result + result1  )
            final = np.where( (final != 0) & (result2 != 0), ((final + result2) /2), final + result2 )
        else:

            d3,final = vid3.read()  
            if d3 == 0:
            	break
            final = cv2.resize(final, (1920,1080), interpolation = cv2.INTER_AREA)/255
            if frames != 0 and counter > frames:
                mask = np.zeros((1080, 1920, 3)).astype(np.float32)
                count += 1
                for x in range(count,count+frames):
                    points = data[x].split("/")
                    for p in points:
                        number += 1
                        if p != '':
                            xandy = p.split(",")
                            p_after = (int(xandy[0]),int(xandy[1]))
                            extract = mask[int(p_after[1]-k/2):int(p_after[1]+k/2),int(p_after[0]-k/2):int(p_after[0]+k/2),:]
                            
                            if(extract.shape[0:2] == gauss.shape):
                                extract = cv2.cvtColor(extract, cv2.COLOR_BGR2GRAY)
                                add = (gauss + extract)/2                                               
                                add = cv2.applyColorMap( ((add)*255).astype(np.uint8),cv2.COLORMAP_JET).astype(np.float32)/255
                                mask[int(p_after[1]-k/2):int(p_after[1]+k/2),int(p_after[0]-k/2):int(p_after[0]+k/2),:] = add
            else: 
                points = data[counter].split("/")
                for p in points:
                    number += 1
                    if p != '':
                        xandy = p.split(",")
                        p_after = (int(xandy[0]),int(xandy[1]))
                        extract = mask[int(p_after[1]-k/2):int(p_after[1]+k/2),int(p_after[0]-k/2):int(p_after[0]+k/2),:]
                        
                        if(extract.shape[0:2] == gauss.shape):
                            extract = cv2.cvtColor(extract, cv2.COLOR_BGR2GRAY)
                            add = (gauss + extract)/2                                         
                            add = cv2.applyColorMap( ((add)*255).astype(np.uint8),cv2.COLORMAP_JET).astype(np.float32)/255
                            mask[int(p_after[1]-k/2):int(p_after[1]+k/2),int(p_after[0]-k/2):int(p_after[0]+k/2),:] = add
                        
        m = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        mas = np.where(m > 0.2 , 1 , 0).astype(np.float32)
        mask_inverse = np.ones(mask.shape)*(1-mas)[:,:,None]
        mask = mask*(mas)[:,:,None]

        final = mask_inverse*final
        final = final + mask

        final = cv2.resize(final, dim, interpolation = cv2.INTER_AREA)

        
        cv2.imshow("Final",final)

        final = final*255
        out.write(final.astype( np.uint8 ))
        
        key=cv2.waitKey(1)
        if key == 27:
            break
    else:
        print("Could not read Video")
        break

file.close()
vid1.release()
vid.release()
vid.release()
out.release()
cv2.destroyAllWindows()