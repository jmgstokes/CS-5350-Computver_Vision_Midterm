#import libaries
import cv2

#Opens video file and writes frames to folder
cap = cv2.VideoCapture('input/input0.avi')
i=0
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == False:
		break
	cv2.imwrite('output/output0_'+str(i)+'.jpg',frame)
	i+=1

cap.release()
cv2.destroyAllWindows()