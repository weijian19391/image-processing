from math import sqrt
import time
import cv2
import numpy as np
import ast
def distance(p0, p1):
	    return ((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)**0.5
def connectDots(frame):
		gaus_blur = cv2.GaussianBlur(frame.astype(np.uint8), (5,5), 0)
		# print type(np.where(gaus_blur < 20, 0, 255))
		# print type(gaus_blur)
		# thresholding =  np.where(gaus_blur < 20, 0, 255).astype(np.uint8)    
		# print thresholding[0]
		_,hahaha = cv2.threshold(gaus_blur, 20, 255, cv2.THRESH_BINARY) # to connect more points of the player
		# print hahaha[0]
		# print np.equal(hahaha, thresholding)
		final_frame = cv2.medianBlur(hahaha, 7) 
		return final_frame

# t0 = time.clock()
# for i in range(0,10):
# 	connectDots(cv2.imread("3148788-1506178378-PR-th.jpg"))
# print "Time taken to get red player position " + str(time.clock() - t0)

fo = open('lalala.txt', 'r')
fo.seek(0, 2)
# print fo.readline()
line = ast.literal_eval(fo.readline())
fo.close()