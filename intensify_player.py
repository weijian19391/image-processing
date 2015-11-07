import cv2
import numpy as np
# np.set_printoptions(threshold='nan')
MIN_AREA_OF_CONTOUR_BOX = 250
RED_THRESHOLD = 80
winname = "GRS"
#bgs_mog = cv2.BackgroundSubtractorMOG(1000, 6, 0.9, 0)
fgbg = cv2.BackgroundSubtractorMOG()
capture = cv2.VideoCapture('football_right.mp4')
# capture = cv2.VideoCapture('input.avi')

j = 0

if __name__ == "__main__":
	while capture.isOpened():
		_,frame = capture.read()
		numRow, numCol, numRGB = frame.shape
		blackAndWhite = np.ndarray((numRow,numCol))
		##### TO INTENSIFY THE COLOURS OF THE PLAYERS
		image_blue = frame[:,:,0].flatten()
		image_green = frame[:,:,1].flatten()
		image_red = frame[:,:,2].flatten()
		cmax = np.maximum(np.maximum(image_blue,image_green), image_red)

		masked_unmax_image_red = np.ma.masked_where(cmax!=image_red, image_red) #red pixels w max values
		masked_lowerintensity_image_red = np.ma.masked_where(masked_unmax_image_red<= RED_THRESHOLD, masked_unmax_image_red)
		max_intensity_image_red = masked_lowerintensity_image_red/masked_lowerintensity_image_red * 255
		# max_intensity_image_red.mask = np.ma.nomask

		max_intensity_image_red_nomask = np.ma.filled(max_intensity_image_red,0)

		final_red = max_intensity_image_red_nomask.reshape(numRow,numCol)
		cv2.imwrite("Images\intensified frame "+ str(j)+ ".png",final_red)
		contours, hierarchy = cv2.findContours(final_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		for i in xrange(0,len(contours)):
			cnt = contours[i]
			rect = cv2.minAreaRect(cnt)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)
			if cv2.contourArea(cnt) > MIN_AREA_OF_CONTOUR_BOX:
				cv2.drawContours(frame,[box],0,(255,255,255),2)
				bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
				cv2.circle(frame, bottommost, 3, (0,0,255),-1)
				# print bottommost

		cv2.imwrite("Images\contoured intensified frame "+ str(j)+ ".png",frame)

		j +=1

		if j == 100:
			exit()


	#cv2.destroyAllWindows()


















