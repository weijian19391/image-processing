import numpy as np
import cv2
import cv2.cv as cv

backgrd = cv2.imread("normImg.png", 1)
backgrd_blue = backgrd[:,:,0].astype(float)
backgrd_green = backgrd[:,:,1].astype(float)
backgrd_red = backgrd[:,:,2].astype(float)
cap = cv2.VideoCapture('football_right.mp4')
width, height, fps, frames_count = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT),  cap.get(cv.CV_CAP_PROP_FPS), cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

frames_count = int(frames_count)


for fr in range(1,10):
	_,frame = cap.read()
	image_blue = frame[:,:,0].astype(float)
	image_green = frame[:,:,1].astype(float)
	image_red = frame[:,:,2].astype(float)
	diff_red = np.absolute(image_red - backgrd_red)
	diff_blue = np.absolute(image_blue - backgrd_blue)
	diff_green = np.absolute(image_green - backgrd_green)
	
	backgrd_mask = (1/3.0*diff_red + 1/3.0*diff_blue + 1/3.0*diff_green).astype(np.uint8)

	_,foreground = cv2.threshold(backgrd_mask, 15, 255, cv2.THRESH_BINARY)
	three_chan_fore = np.dstack((foreground,foreground,foreground))
	colour_foregrd = cv2.bitwise_and(three_chan_fore, frame)
	cv2.imwrite("Foreground\lalarocks"+str(fr)+".png", colour_foregrd)
