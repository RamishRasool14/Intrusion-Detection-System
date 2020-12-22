import cv2 
from time import time

video1 = cv2.VideoCapture("http://192.168.8.116:8080/video") 
video2 = cv2.VideoCapture("http://192.168.8.117:8080/video") 
video3 = cv2.VideoCapture(0) 
length = int(input("Seconds to record: "))
check = int(input("Press 1 to show live feed else 0 "))
frame1_width = int(video1.get(3)) 
frame1_height = int(video1.get(4))    
size1 = (frame1_width, frame1_height) 

frame2_width = int(video2.get(3)) 
frame2_height = int(video2.get(4))    
size2 = (frame2_width, frame2_height) 

frame3_width = int(video3.get(3)) 
frame3_height = int(video3.get(4))    
size3 = (frame3_width, frame3_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output  
# is stored in 'filename.avi' file. 
result1 = cv2.VideoWriter('Live1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size1)
result2 = cv2.VideoWriter('Live2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size2)
result3 = cv2.VideoWriter('Live3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size3)

start = time()

  
while(True): 

	ret1, frame1 = video1.read() 
	ret2, frame2 = video2.read() 
	ret3, frame3 = video3.read() 
    
	# assert ret1
	# assert ret2
	# assert ret3


	# if ret3 == True:

	if check == 1:

		cv2.imshow('Live1', frame1) 
		cv2.imshow('Live2', frame2) 
		cv2.imshow('Live3', frame3) 


	result1.write(frame1)
	result2.write(frame2)
	result3.write(frame3)
	end = time()

	if ( end - start ) > length: 
		break
	# else:
	# 	break
  
video1.release()
video2.release()
video3.release()

result1.release()
result2.release()
result3.release()
cv2.destroyAllWindows() 